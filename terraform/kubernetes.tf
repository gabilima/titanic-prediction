# Kubernetes cluster configuration

# Create EKS cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 18.0"

  cluster_name    = "${var.project_name}-${var.environment}-cluster"
  cluster_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Enable private access to the cluster
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true

  # Cluster security group
  create_cluster_security_group = true
  create_node_security_group    = true

  # Node groups configuration
  eks_managed_node_groups = {
    main = {
      name           = "main-node-group"
      instance_types = var.node_instance_types
      min_size       = var.node_min_size
      max_size       = var.node_max_size
      desired_size   = var.node_desired_capacity

      # Enable autoscaling
      create_iam_role          = true
      iam_role_name            = "${var.project_name}-${var.environment}-eks-node-role"
      iam_role_use_name_prefix = false
      iam_role_description     = "EKS managed node group role"
      iam_role_tags = {
        Purpose = "EKS managed node group role"
      }
      iam_role_additional_policies = [
        "arn:aws:iam::aws:policy/AmazonECR-FullAccess",
        "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
      ]

      # Block device configuration
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 50
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }

      # Tags
      tags = {
        Environment = var.environment
        NodeGroup   = "main"
      }
    }
  }

  # Cluster access entry configuration
  manage_aws_auth_configmap = true
  aws_auth_roles = [
    {
      rolearn  = module.eks.eks_managed_node_groups["main"].iam_role_arn
      username = "system:node:{{EC2PrivateDNSName}}"
      groups   = ["system:bootstrappers", "system:nodes"]
    }
  ]

  # Cluster tags
  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Deploy Metrics Server
resource "helm_release" "metrics_server" {
  name       = "metrics-server"
  repository = "https://kubernetes-sigs.github.io/metrics-server/"
  chart      = "metrics-server"
  namespace  = "kube-system"
  version    = "3.8.2"

  depends_on = [module.eks]
}

# Deploy Prometheus for monitoring
resource "helm_release" "prometheus" {
  name       = "prometheus"
  repository = "https://prometheus-community.github.io/helm-charts"
  chart      = "kube-prometheus-stack"
  namespace  = "monitoring"
  version    = "42.0.3"
  create_namespace = true

  values = [<<EOF
grafana:
  adminPassword: "${var.environment}-admin-password" # Use AWS Secrets Manager in production
  persistence:
    enabled: true
    size: 10Gi
  service:
    type: ClusterIP
  ingress:
    enabled: true
    annotations:
      kubernetes.io/ingress.class: nginx
      cert-manager.io/cluster-issuer: letsencrypt
    hosts:
      - grafana.${var.project_name}.${var.environment}.example.com
    tls:
      - secretName: grafana-

