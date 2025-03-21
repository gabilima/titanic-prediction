# Output definitions for the Titanic prediction service infrastructure

# EKS Cluster Information
output "eks_cluster_id" {
  description = "The name of the EKS cluster"
  value       = module.eks.cluster_id
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "eks_config_map_aws_auth" {
  description = "A kubernetes configuration to authenticate to this EKS cluster"
  value       = module.eks.config_map_aws_auth
  sensitive   = true
}

output "eks_cluster_certificate_authority_data" {
  description = "Certificate authority data for the EKS cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

# Redis Feature Store Endpoint
output "redis_endpoint" {
  description = "The endpoint of the Redis cluster for feature store"
  value       = module.redis.endpoint
}

output "redis_port" {
  description = "The port of the Redis cluster"
  value       = module.redis.port
}

output "redis_connection_string" {
  description = "Connection string for Redis"
  value       = "redis://${module.redis.endpoint}:${module.redis.port}"
  sensitive   = true
}

# MLflow Database Connection
output "mlflow_db_endpoint" {
  description = "The endpoint of the MLflow database"
  value       = module.rds.endpoint
}

output "mlflow_db_name" {
  description = "The name of the MLflow database"
  value       = module.rds.db_name
}

output "mlflow_db_user" {
  description = "The username for the MLflow database"
  value       = module.rds.username
  sensitive   = true
}

output "mlflow_db_connection_string" {
  description = "Connection string for MLflow database"
  value       = "postgresql://${module.rds.username}:${module.rds.password}@${module.rds.endpoint}:5432/${module.rds.db_name}"
  sensitive   = true
}

output "mlflow_service_endpoint" {
  description = "The endpoint for the MLflow tracking server"
  value       = "http://${kubernetes_service.mlflow.status.0.load_balancer.0.ingress.0.hostname}"
}

# VPC and Subnet Information
output "vpc_id" {
  description = "The ID of the VPC"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "database_subnets" {
  description = "List of IDs of database subnets"
  value       = module.vpc.database_subnets
}

output "nat_gateway_ips" {
  description = "List of allocation IDs of the NAT gateways"
  value       = module.vpc.nat_public_ips
}

# Security Group IDs
output "app_security_group_id" {
  description = "The ID of the application security group"
  value       = aws_security_group.app_sg.id
}

output "db_security_group_id" {
  description = "The ID of the database security group"
  value       = aws_security_group.db_sg.id
}

output "redis_security_group_id" {
  description = "The ID of the Redis security group"
  value       = aws_security_group.redis_sg.id
}

output "monitoring_security_group_id" {
  description = "The ID of the monitoring security group"
  value       = aws_security_group.monitoring_sg.id
}

# Monitoring Endpoints
output "prometheus_endpoint" {
  description = "The endpoint for the Prometheus server"
  value       = "http://${kubernetes_service.prometheus.status.0.load_balancer.0.ingress.0.hostname}"
}

output "grafana_endpoint" {
  description = "The endpoint for the Grafana dashboard"
  value       = "http://${kubernetes_service.grafana.status.0.load_balancer.0.ingress.0.hostname}"
}

output "alertmanager_endpoint" {
  description = "The endpoint for the Alertmanager"
  value       = "http://${kubernetes_service.alertmanager.status.0.load_balancer.0.ingress.0.hostname}"
}

# API Information
output "api_endpoint" {
  description = "The endpoint for the prediction API"
  value       = "https://${kubernetes_ingress.api_ingress.status.0.load_balancer.0.ingress.0.hostname}"
}

# ECR Repository
output "ecr_repository_url" {
  description = "The URL of the ECR repository for the application"
  value       = aws_ecr_repository.app_repository.repository_url
}

# S3 Bucket for Model Artifacts
output "model_artifacts_bucket" {
  description = "The name of the S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

# IAM Roles
output "eks_worker_role_arn" {
  description = "ARN of the EKS worker nodes IAM role"
  value       = module.eks.worker_iam_role_arn
}

output "mlflow_role_arn" {
  description = "ARN of the IAM role for MLflow"
  value       = aws_iam_role.mlflow_role.arn
}

