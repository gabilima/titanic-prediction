# General variables
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "titanic-ml"
}

# Network variables
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

# EKS variables
variable "cluster_version" {
  description = "Kubernetes version to use for EKS cluster"
  type        = string
  default     = "1.24"
}

variable "node_instance_types" {
  description = "EC2 instance types for node groups"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_desired_capacity" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 2
}

variable "node_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 5
}

# Redis variables
variable "redis_node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.small"
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes in Redis cluster"
  type        = number
  default     = 1
}

# MLflow variables
variable "mlflow_db_instance_class" {
  description = "Database instance class for MLflow"
  type        = string
  default     = "db.t3.small"
}

variable "mlflow_s3_bucket_name" {
  description = "S3 bucket name for MLflow artifacts"
  type        = string
  default     = "titanic-ml-mlflow-artifacts"
}

