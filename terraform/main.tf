# Main infrastructure configuration

# Create VPC for the cluster
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "${var.project_name}-${var.environment}-vpc"
  cidr = var.vpc_cidr

  azs             = var.availability_zones
  private_subnets = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i)]
  public_subnets  = [for i, az in var.availability_zones : cidrsubnet(var.vpc_cidr, 8, i + 100)]

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment != "prod"
  one_nat_gateway_per_az = var.environment == "prod"
  enable_vpn_gateway     = false

  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags required for EKS
  private_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-${var.environment}-cluster" = "shared"
    "kubernetes.io/role/internal-elb"                                      = "1"
  }

  public_subnet_tags = {
    "kubernetes.io/cluster/${var.project_name}-${var.environment}-cluster" = "shared"
    "kubernetes.io/role/elb"                                                = "1"
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
  }
}

# Create security groups
resource "aws_security_group" "mlflow" {
  name        = "${var.project_name}-${var.environment}-mlflow-sg"
  description = "Security group for MLflow server"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "MLflow server"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-mlflow-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "redis" {
  name        = "${var.project_name}-${var.environment}-redis-sg"
  description = "Security group for Redis feature store"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
    description = "Redis"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name        = "${var.project_name}-${var.environment}-redis-sg"
    Environment = var.environment
  }
}

# Create S3 bucket for MLflow artifacts
resource "aws_s3_bucket" "mlflow_artifacts" {
  bucket = "${var.mlflow_s3_bucket_name}-${var.environment}"

  tags = {
    Name        = "${var.mlflow_s3_bucket_name}-${var.environment}"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow_artifacts" {
  bucket = aws_s3_bucket.mlflow_artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Create RDS instance for MLflow backend
resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.project_name}-${var.environment}-mlflow-subnet-group"
  subnet_ids = module.vpc.private_subnets

  tags = {
    Name        = "${var.project_name}-${var.environment}-mlflow-subnet-group"
    Environment = var.environment
  }
}

resource "aws_db_instance" "mlflow" {
  identifier             = "${var.project_name}-${var.environment}-mlflow-db"
  engine                 = "postgres"
  engine_version         = "13.7"
  instance_class         = var.mlflow_db_instance_class
  allocated_storage      = 20
  db_name                = "mlflow"
  username               = "mlflow"
  password               = "mlflow_password" # Use AWS Secrets Manager in production
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  vpc_security_group_ids = [aws_security_group.mlflow.id]
  skip_final_snapshot    = true
  storage_encrypted      = true

  tags = {
    Name        = "${var.project_name}-${var.environment}-mlflow-db"
    Environment = var.environment
  }
}

# Create ElastiCache Redis cluster for feature store
resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-${var.environment}-redis-subnet-group"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "${var.project_name}-${var.environment}-redis"
  engine               = "redis"
  node_type            = var.redis_node_type
  num_cache_nodes      = var.redis_num_cache_nodes
  parameter_group_name = "default.redis6.x"
  engine_version       = "6.2"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.redis.name
  security_group_ids   = [aws_security_group.redis.id]

  tags = {
    Name        = "${var.project_name}-${var.environment}-redis"
    Environment = var.environment
  }
}

