# Deployment Guide

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Model Storage & S3 Setup](#model-storage--s3-setup)
4. [Cloud Deployment](#cloud-deployment)
5. [Production Considerations](#production-considerations)

## Local Deployment

### Prerequisites
- Python 3.8+
- Virtual environment
- 8GB+ RAM (16GB recommended for transformer models)

### Steps

```bash
# 1. Clone repository
git clone https://github.com/irfanfetahovic/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download trained model (if not training from scratch)
# Place model files in models/distilbert_sentiment/ or models/classical_models

# 5. Run API server
# FastAPI (default):
python app/app_fastapi.py

# Or Flask alternative:
python app/app.py
```

API will be available at `http://localhost:5000`

## Docker Deployment

### Image Size Note

The Docker image is approximately **3.5GB** due to PyTorch and Transformers dependencies. This is typical for ML inference containers. For this reason:
- **CI/CD builds the image but does not push to a registry** (to avoid slow pushes/pulls)
- For local development/demo, build the image locally
- For production, consider using ONNX Runtime to reduce size to ~700MB

### Using a Container Registry (Optional)

If you want to push to a registry, you have several options:

**GitHub Container Registry (GHCR):**
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Tag and push
docker tag sentiment-analysis:latest ghcr.io/YOUR_USERNAME/sentiment-analysis:latest
docker push ghcr.io/YOUR_USERNAME/sentiment-analysis:latest

# Pull from GHCR
docker pull ghcr.io/YOUR_USERNAME/sentiment-analysis:latest
```

**Docker Hub:**
```bash
# Login to Docker Hub
docker login -u YOUR_USERNAME

# Tag and push
docker tag sentiment-analysis:latest YOUR_USERNAME/sentiment-analysis:latest
docker push YOUR_USERNAME/sentiment-analysis:latest
```

> **Note:** Large images (~3.5GB) will take significant time to push/pull. Consider this for CI/CD pipeline duration.

### Build Optimization

The project includes a `.dockerignore` file that excludes unnecessary files (tests, notebooks, docs, large data files) to:
- Reduce build context size (~100x smaller)
- Speed up builds (30-60 seconds faster)
- Prevent build timeouts

### Build and Run

```bash
# Build Docker image
docker build -t sentiment-analysis:latest .

# Run container ("%cd% for CMD use, ${PWD} for Power Shell)
docker run -d \
  -p 5000:5000 \
  -v $(PWD)/models:/app/models:ro \ 
  --name sentiment-api \
  sentiment-analysis:latest

# Check logs
docker logs -f sentiment-api

# Stop container
docker stop sentiment-api
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Model Storage & S3 Setup

### Overview

This project follows professional ML deployment practices by **separating code from models**:
- **Docker image**: Contains application code only (~300MB)
- **Model files**: Stored externally in S3, GitHub Releases, or similar
- **Benefits**: Faster builds, independent versioning, lower costs, deployment flexibility

The Docker image includes an entrypoint script that automatically downloads models on container startup.

### Model Preparation

Package your trained model as a tar.gz archive:

```bash
# From project root
cd models
tar -czf distilbert_sentiment.tar.gz distilbert_sentiment/
tar -czf classical_models.tar.gz classical_models/


# Verify archive contents
tar -tzf distilbert_sentiment.tar.gz
```

The archive should contain:
```
distilbert_sentiment/
├── config.json
├── model.safetensors
├── vocab.txt
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

### Upload to S3

#### Using AWS Console

1. Go to S3 → Create bucket (e.g., `irfan-ml-models`)
2. Upload `distilbert_sentiment.tar.gz` to desired path
3. Note the URI: `s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz`

#### Using AWS CLI

```bash
# Create bucket (one time)
aws s3 mb s3://irfan-ml-models

# Upload model
aws s3 cp models/distilbert_sentiment.tar.gz \
  s3://irfan-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz

aws s3 cp models/classical_models.tar.gz \
  s3://irfan-ml-models/sentiment-analysis/classical_models.tar.gz

# Verify upload
aws s3 ls s3://my-ml-models/sentiment-analysis/

# Set private access (recommended)
aws s3api put-object-acl \
  --bucket my-ml-models \
  --key sentiment-analysis/distilbert_sentiment.tar.gz \
  --acl private
```

### IAM Permissions

#### Option A: IAM Role (Recommended for AWS deployments)

Example for creating an IAM role with S3 read permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-ml-models/*",
        "arn:aws:s3:::my-ml-models"
      ]
    }
  ]
}
```

Attach this role to your ECS task definition or EC2 instance profile.

#### Option B: Access Keys (For non-AWS deployments)

Create an IAM user with the above policy, generate access keys, and use as environment variables.

### Container Deployment with S3 Models

#### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_S3_URI` | No* | S3 URI for model (e.g., `s3://bucket/path/model.tar.gz`) |
| `MODEL_URL` | No* | HTTP URL for model (GitHub Releases, etc.) |
| `AWS_ACCESS_KEY_ID` | If using S3 without IAM | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | If using S3 without IAM | AWS secret key |
| `AWS_DEFAULT_REGION` | If using S3 | AWS region (e.g., `us-east-1`) |

*Set either `MODEL_S3_URI` or `MODEL_URL`, not both.

#### Local Testing

```bash
docker run -d \
  -p 5000:5000 \
  -e MODEL_S3_URI=s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz \
  -e AWS_ACCESS_KEY_ID=YOUR_KEY \
  -e AWS_SECRET_ACCESS_KEY=YOUR_SECRET \
  -e AWS_DEFAULT_REGION=us-east-1 \
  sentiment-analysis:latest
```

#### Using GitHub Releases (Alternative to S3)

If you prefer not to use S3:

```bash
# Create release and upload model
gh release create v1.0.0 models/distilbert_sentiment.tar.gz

# Deploy with MODEL_URL
docker run -d \
  -p 5000:5000 \
  -e MODEL_URL=https://github.com/your-username/sentiment-analysis/releases/download/v1.0.0/distilbert_sentiment.tar.gz \
  sentiment-analysis:latest
```

### How It Works

1. Container starts → Entrypoint script executes
2. Checks if `/app/models/distilbert_sentiment` exists
3. If missing and `MODEL_S3_URI` or `MODEL_URL` is set → Downloads model
4. Extracts model to `/app/models/`
5. Starts gunicorn application server
6. Model remains cached for container lifetime (or in persistent volume)

### Cost Considerations

**S3 Storage:**
- Standard tier: ~$0.023/GB/month
- Example: 500MB model = ~$0.01/month

**Data Transfer:**
- Within AWS (same region): Free
- AWS to internet: $0.09/GB after 100GB/month free tier
- Example: 500MB model × 100 container restarts/month = $4.50

**Optimization:** Use persistent volumes (EFS, EBS) to cache models and minimize S3 downloads.

### Troubleshooting

**Model download fails:**

```bash
# Check container logs
docker logs <container-id>

# Verify S3 access from container
docker exec -it <container-id> aws s3 ls s3://my-ml-models/

# Test manual download
docker exec -it <container-id> bash
aws s3 cp $MODEL_S3_URI /tmp/test.tar.gz
```

**Slow startup / health check fails:**

- Model download adds 5-30 seconds to startup time
- Health check `start-period` is set to 90s to accommodate downloads
- Increase if your model is larger: Edit `HEALTHCHECK --start-period=120s` in Dockerfile

**Container starts but predictions fail:**

```bash
# Check if model was downloaded
docker exec -it <container-id> ls -la /app/models/distilbert_sentiment/

# Check environment variables
docker exec -it <container-id> env | grep MODEL
```

## Cloud Deployment

### AWS Deployment

#### AWS ECS (Fargate)

**Prerequisites**: Create ECR repository first (via AWS Console or CLI):

```bash
aws configure # login into IAM user which contain necessary privilages
# needed: aws configure command and setting up aws cli user
# accound-id is 12-digit number and can be found with aws cli command "aws sts get-caller-identity"
# aws cli user should have also AmazonEC2ContainerRegistryPowerUser and AmazonEC2ContainerRegistryFullAccess policy
aws ecr create-repository --repository-name sentiment-analysis
```

**Build and Push to ECR:**

```bash
# 1. Build Docker image locally
docker build -t sentiment-analysis:latest .

# 2. Authenticate with ECR

aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

# 3. Tag for ECR
docker tag sentiment-analysis:latest <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest

# 4. Push to ECR
docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
```

<!-- Alternative: If you push to GHCR, you can pull from there instead:
# docker pull ghcr.io/<your-github-username>/sentiment-analysis:latest
# docker tag ghcr.io/<your-github-username>/sentiment-analysis:latest <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
# docker push <account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest
-->

**Next Steps**: Create ECS task definition and service

##### 1. Create ECS Task definitions

You must register a task definition before creating a service. Below is a professional, secure sample for this project:

> **Security Best Practice:**
> Do NOT include AWS credentials or secrets directly in your task definition or version control. Use IAM roles for tasks (recommended) or reference secrets from AWS Secrets Manager or SSM Parameter Store.

1. Create a file named `ecs-task-definition.json` in your project root with the following content (replace `<account-id>`, `<region>`, and ARNs as needed):

```json
{
  "family": "sentiment-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::<account-id>:role/sentimentAppTaskRole",
  "containerDefinitions": [
    {
      "name": "sentiment-api",
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/sentiment-analysis:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        { "name": "MODEL_S3_URI", "value": "s3://my-ml-models/sentiment-analysis/distilbert_sentiment.tar.gz" },
        { "name": "AWS_DEFAULT_REGION", "value": "<region>" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analysis",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:5000/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 90
      }
    }
  ]
}
```

- `executionRoleArn` and `taskRoleArn` should be IAM roles with permissions to pull from ECR, write logs, and access S3/SSM as needed.


2. Register the task definition:

```bash
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json
```

After successful registration, use the outputted task definition revision (e.g., sentiment-analysis:1) in the next step.

##### 2. Create ECS service

First create a cluster

aws ecs create-cluster `
  --region eu-north-1 `
  --cluster-name sentiment-cluster

Then choose subnet and security group, and create log group

List of subnets to choose from
aws ec2 describe-subnets --region eu-north-1 --query "Subnets[*].{ID:SubnetId,Name:Tags[?Key=='Name']|[0].Value}" 

List of security groups to choose from
aws ec2 describe-security-groups --region eu-north-1 --query "SecurityGroups[*].{ID:GroupId,Name:GroupName}"


aws logs create-log-group --log-group-name /ecs/sentiment-analysis --region eu-north-1

Create service

Replace subnet-XXXX and sg-YYYY with the IDs you found in the first two commands in the previous section.

aws ecs create-service `
  --region eu-north-1 `
  --cluster sentiment-cluster `
  --service-name sentiment-service `
  --task-definition sentiment-analysis:1 `# 1 is a revision number, if ommitted aws uses last version
  --desired-count 1 `
  --launch-type FARGATE `
  --network-configuration "awsvpcConfiguration={subnets=[subnet-XXXX],securityGroups=[sg-YYYY],assignPublicIp=ENABLED}"
  Note: This service does not have load balacing and autoscaling, but it can be configured.

##### 3. Useful commands

aws ecs describe-services --cluster sentiment-cluster --services sentiment-service --region eu-north-1
aws ecs list-tasks --cluster sentiment-cluster --service-name sentiment-service --region eu-north-1 # if not empty, it is good
aws ecs describe-tasks --cluster sentiment-cluster --tasks xxxxxxxxxxxxxxxxxxxx --region eu-north-1
aws ecs delete-service --cluster sentiment-cluster --service sentiment-service --force --region eu-north-1
aws ecs update-service --cluster sentiment-cluster --service sentiment-service --force-new-deployment --region eu-north-1


##### 4. Testing 

aws ecs list-tasks --cluster sentiment-cluster --service-name sentiment-service --region eu-north-1 # if not empty, it is good
aws ecs describe-tasks --cluster sentiment-cluster --tasks xxxxxxxxxxxxxxxxxxxx --region eu-north-1

Security Group Rules (EC2-Security Groups)
Go to the security group sg-0a213725b35d2fe47 (the one attached to your Fargate task) in the EC2 console:
Inbound rule: Allow TCP on port 5000 from your IP (My IP) or 0.0.0.0/0 (for testing, less secure).

Outbound rule: Usually default allows all outbound, which is fine.

Go to VPC-Subnets-Select subnet you previously selected in "List of subnets to choose from" and check if Auto-assign customer-owned IPv4 address is Yes. If not, go to Actions and edit this.

curl http://<PUBLIC_IP>:5000/health
Public IP is available at ECS-Clusters-<our cluster>-Task, then click on active task and select Network-ENI-Network Interface

Stoping tasks (and stop incurring charges by AWS), but service remain and can be restarted
aws ecs update-service --cluster sentiment-cluster --service sentiment-service --desired-count 0 --region eu-north-1
Restarting
aws ecs update-service --cluster sentiment-cluster --service sentiment-service --desired-count 1 --region eu-north-1

### DigitalOcean

#### Droplets with Docker

```bash
# 1. Create Droplet with Docker pre-installed
First, install doctl drom DigitalOcean website, and generate a Digital Ocean API token.
Issue commmand doctl auth init and paste the token.
List your SSH keys (get the ID): doctl compute ssh-key list
If no keys, generate one 
ssh-keygen -t ed25519 -C "irfanfetahovic@gmail.com" -f $env:USERPROFILE\.ssh\id_ed25519_do
Import public key into DigitalOcean doctl compute ssh-key import do-key `
  --public-key-file $env:USERPROFILE\.ssh\id_ed25519_do.pub

Run doctl compute ssh-key list

doctl compute droplet create sentiment-api `# Droplet is a VM on Digital Ocean
  --image docker-20-04 ` # Ubuntu 20.04 with Docker pre-installed
  --size s-2vcpu-4gb `
  --region nyc3 `
  --ssh-keys <ID_FROM_LIST>

doctl compute droplet list # Useful command to get droplet IP address

# 2. SSH into droplet
doctl compute ssh sentiment-api
Alternative: doctl compute ssh -i $env:USERPROFILE\.ssh\id_ed25519_do root@<dropletIP>


# 3. Build image on droplet (or pull from registry)
git clone https://github.com/<your-username>/sentiment-analysis.git
cd sentiment-analysis
docker build -t sentiment-analysis:latest .

# Alternative: pull from GHCR if you push there
# docker pull ghcr.io/<your-username>/sentiment-analysis:latest

# 4. Run container

docker run -d \
  -p 80:5000 \
  --name sentiment-api \
  --restart unless-stopped \
  -e AWS_ACCESS_KEY_ID=<your-access-key> \
  -e AWS_SECRET_ACCESS_KEY=<your-secret-key> \
  -e AWS_DEFAULT_REGION=<your-region> \
  -e TRANSFORMER_MODEL_S3_URI=s3://your-bucket-name/model-file \
   -e CLASSICAL_MODEL_S3_URI=s3://your-bucket-name/model-file \
  sentiment-analysis:latest

Check if its working: docker ps
STATUS should say Up
PORTS shows host_port -> container_port (here 80->5000)
docker logs -f sentiment-api

If not working try again, but first stop and remove container that is not working
docker stop <container-id> # Stopping container
docker rm <container-id> # Removing container

exit

# 4. Configure firewall (allow HTTP)
doctl compute firewall create \
  --name web-firewall \
  --inbound-rules "protocol:tcp,ports:80,sources:addresses:0.0.0.0/0,sources:addresses:::/0" \
  --droplet-ids $(doctl compute droplet list sentiment-api --format ID --no-header)
```
5. Testing
curl http://<droplet-public-ip>/
http://<droplet-public-ip>/
http://<droplet-public-ip>/docs

Useful commands
Before accessing droplet you have to know its ip address
doctl compute droplet list # To see droplet ip address and other info


doctl compute firewall list # firewall info
ssh root@<droplet-public-ip> or ssh -i $env:USERPROFILE\.ssh\id_ed25519_do root@167.71.172.85 # connecting to droplet again after you exit

doctl compute firewall update f50d4ffe-3b27-41e8-9bc1-baea6c6ea0e4 # Firewall ID`
  --name web-firewall `
  --inbound-rules "protocol:tcp,ports:22,sources:addresses:0.0.0.0/0,sources:addresses:::/0" `
  --inbound-rules "protocol:tcp,ports:80,sources:addresses:0.0.0.0/0,sources:addresses:::/0"
This command updates firewall rules to include port 22 necessary for ssh access.

Stopping droplet
doctl compute droplet-action power-off <droplet-id>
Cost: You still pay for the droplet disk (storage), but not for CPU/RAM while it’s powered off
Useful if you want to pause usage but keep the data and configuration

Deleting droplet
doctl compute droplet delete <droplet-id> --force
Cost: You stop all charges, but all data is lost unless you have snapshots/backups


## Production Considerations

### CI/CD Pipeline

GitHub Actions (see `.github/workflows/ci.yaml`) provides:
- **Automated testing** across Python 3.10, 3.11, 3.12
- **Docker image building** (build verification only, no push):
  - GitHub Actions cache (`type=gha,mode=max`) for 50-80% faster rebuilds
  - Verifies Dockerfile builds correctly on every push to main
- **Build efficiency**: `.dockerignore` reduces context size significantly

> **Note**: The image (~3.5GB) is not pushed to a registry by default due to size. 
> To enable GHCR push, uncomment the relevant section in `ci.yaml`. 
> See [Using a Container Registry](#using-a-container-registry-optional) for manual push options.

### Cost Optimization

#### AWS
- Use Spot Instances for training
- S3 for model storage
- Lambda for low-traffic APIs

#### General
- Implement request batching
- Use model quantization
- Cache frequent predictions
- Monitor and optimize resource usage

## Testing Deployment

```bash
# Health check
curl http://your-deployment-url/health

# Test prediction
curl -X POST http://your-deployment-url/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'

```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, use quantized model
2. **Slow Inference**: Use GPU, optimize model, implement caching
3. **Container crashes**: Check resource limits, review logs
4. **Model not loading**: Verify model path, check file permissions

### Debug Commands

```bash
# Check container logs
docker logs sentiment-api

# Exec into container
docker exec -it sentiment-api bash

# Check resource usage
docker stats sentiment-api

# Test locally
python -c "from src.inference import SentimentPredictor; p = SentimentPredictor('models/distilbert_sentiment'); print(p.predict('test'))"
```
