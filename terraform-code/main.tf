terraform {
  backend "s3" {
    bucket         = "terraform-work-studies"  # 🔹 Create an S3 bucket for this
    key            = "terraform.tfstate"
    region         = "us-east-1"  # 🔹 Change to your region
    encrypt        = true
    dynamodb_table = "terraform-lock-table"  # 🔹 Create a DynamoDB table for locking
  }
}

provider "aws" {
  region     = var.aws_region
  access_key = var.aws_access_key
  secret_key = var.aws_secret_key
}

variable "aws_region" {
  description = "AWS region"
  type        = string
}

variable "aws_access_key" {
  description = "AWS access key"
  type        = string
}

variable "aws_secret_key" {
  description = "AWS secret key"
  type        = string
}

# ✅ SSH Key Pair for EC2
resource "aws_key_pair" "my_key" {
  key_name   = "my-terraform-key"
  public_key = file("${path.module}/my-terraform-key.pub")
}

# ✅ VPC (Virtual Private Cloud)
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"

  name    = "flask-vpc"
  cidr    = "10.0.0.0/16"

  azs             = ["us-east-1a", "us-east-1b"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway = false
}

# ✅ Security Group for EC2 & Load Balancer
resource "aws_security_group" "allow_http" {
  vpc_id = module.vpc.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ✅ EC2 Instance (Flask API)
module "ec2_instance" {
  source  = "terraform-aws-modules/ec2-instance/aws"
  version = "5.0.0"

  name                    = "flask-instance"
  ami                     = "ami-0c104f6f4a5d9d1d5"
  instance_type           = "t2.micro"
  subnet_id               = module.vpc.public_subnets[0]
  vpc_security_group_ids  = [aws_security_group.allow_http.id]
  key_name                = aws_key_pair.my_key.key_name  

  associate_public_ip_address = true

  # ✅ Install Flask API Automatically
  user_data = <<-EOF
              #!/bin/bash
              sudo yum update -y
              sudo yum install -y python3 python3-pip
              pip3 install flask

              # Create Flask App
              mkdir -p /home/ec2-user/app
              cat << EOF2 > /home/ec2-user/app/app.py
              from flask import Flask
              app = Flask(__name__)

              @app.route("/")
              def home():
                  return "Hello, Flask is running on EC2!"

              if __name__ == "__main__":
                  app.run(host="0.0.0.0", port=80)
              EOF2

              # Run Flask API
              python3 /home/ec2-user/app/app.py &
              EOF

  tags = {
    Terraform = "true"
    Name      = "FlaskEC2Instance"
  }
}

# ✅ Load Balancer
resource "aws_lb" "flask_lb" {
  name               = "flask-load-balancer"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.allow_http.id]
  subnets            = module.vpc.public_subnets
}

# ✅ Target Group
resource "aws_lb_target_group" "flask_tg" {
  name     = "flask-target-group"
  port     = 80
  protocol = "HTTP"
  vpc_id   = module.vpc.vpc_id

  health_check {
    path                = "/"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 2
  }
}

# ✅ Load Balancer Listener
resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.flask_lb.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.flask_tg.arn
  }
}

# ✅ Attach EC2 to Load Balancer
resource "aws_lb_target_group_attachment" "flask_instance" {
  target_group_arn = aws_lb_target_group.flask_tg.arn
  target_id        = module.ec2_instance.id
  port             = 80
}
