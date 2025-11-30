# AWS EC2 Deployment Guide for Water Level Monitoring App

This guide will walk you through deploying your Streamlit application on AWS EC2.

## Prerequisites

- AWS Account
- AWS CLI installed (optional but recommended)
- SSH client (built into Linux/Mac, use PuTTY or WSL for Windows)
- Basic knowledge of Linux commands

## Step 1: Launch EC2 Instance

### 1.1 Login to AWS Console
1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Navigate to **EC2** service

### 1.2 Launch Instance
1. Click **"Launch Instance"**
2. Configure the instance:
   - **Name**: `water-level-monitoring-app`
   - **AMI**: Choose **Amazon Linux 2023** or **Ubuntu 22.04 LTS** (recommended)
   - **Instance Type**: 
     - For testing: `t2.micro` (Free Tier eligible)
     - For production: `t3.small` or `t3.medium`
   - **Key Pair**: 
     - Create new key pair or use existing
     - Download the `.pem` file (you'll need this to SSH)
   - **Network Settings**: 
     - Allow SSH (port 22) from your IP
     - Allow HTTP (port 80) from anywhere (0.0.0.0/0)
     - Allow HTTPS (port 443) from anywhere (0.0.0.0/0)
     - Allow Custom TCP (port 8501) from anywhere for Streamlit
   - **Storage**: 20 GB should be sufficient
3. Click **"Launch Instance"**

### 1.3 Configure Security Group
1. Go to **Security Groups** in EC2 console
2. Find your instance's security group
3. Add inbound rules:
   - **Type**: Custom TCP
   - **Port**: 8501
   - **Source**: 0.0.0.0/0 (or your specific IP for security)
   - **Description**: Streamlit App

## Step 2: Connect to EC2 Instance

### 2.1 Get Instance Public IP
1. In EC2 console, find your instance
2. Note the **Public IPv4 address**

### 2.2 SSH into Instance

**For Linux/Mac:**
```bash
chmod 400 your-key-pair.pem
ssh -i your-key-pair.pem ec2-user@YOUR_PUBLIC_IP
# For Ubuntu, use: ssh -i your-key-pair.pem ubuntu@YOUR_PUBLIC_IP
```

**For Windows (using WSL or Git Bash):**
```bash
chmod 400 your-key-pair.pem
ssh -i your-key-pair.pem ec2-user@YOUR_PUBLIC_IP
```

**For Windows (using PuTTY):**
1. Convert `.pem` to `.ppk` using PuTTYgen
2. Use PuTTY with the `.ppk` file

## Step 3: Install Dependencies

### 3.1 Update System (Amazon Linux)
```bash
sudo yum update -y
```

### 3.2 Update System (Ubuntu)
```bash
sudo apt update && sudo apt upgrade -y
```

### 3.3 Install Python and pip
**For Amazon Linux:**
```bash
sudo yum install python3 python3-pip -y
```

**For Ubuntu:**
```bash
sudo apt install python3 python3-pip python3-venv -y
```

### 3.4 Install Git
**For Amazon Linux:**
```bash
sudo yum install git -y
```

**For Ubuntu:**
```bash
sudo apt install git -y
```

## Step 4: Clone and Setup Project

### 4.1 Clone Your Repository
```bash
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
# OR upload files using SCP
```

**Alternative: Upload files using SCP (from your local machine)**
```bash
scp -i your-key-pair.pem -r /path/to/your/project ec2-user@YOUR_PUBLIC_IP:~/
```

### 4.2 Navigate to Project Directory
```bash
cd ~/YOUR_PROJECT_DIRECTORY
```

### 4.3 Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.4 Install Python Dependencies
```bash
pip install --upgrade pip
pip install streamlit pandas numpy plotly requests beautifulsoup4 joblib torch torchvision selenium pyyaml lightgbm
```

**Or if you have a requirements.txt:**
```bash
pip install -r requirements.txt
```

### 4.5 Install System Dependencies (for Selenium if needed)
**For Amazon Linux:**
```bash
sudo yum install chromium chromium-headless -y
```

**For Ubuntu:**
```bash
sudo apt install chromium-browser chromium-chromedriver -y
```

## Step 5: Configure Application

### 5.1 Create Systemd Service (Recommended for Production)

Create a service file:
```bash
sudo nano /etc/systemd/system/streamlit-app.service
```

Add the following content:
```ini
[Unit]
Description=Streamlit Water Level Monitoring App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/YOUR_PROJECT_DIRECTORY
Environment="PATH=/home/ec2-user/YOUR_PROJECT_DIRECTORY/venv/bin"
ExecStart=/home/ec2-user/YOUR_PROJECT_DIRECTORY/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Important:** Replace `YOUR_PROJECT_DIRECTORY` with your actual project directory name.

### 5.2 Enable and Start Service
```bash
sudo systemctl daemon-reload
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app
sudo systemctl status streamlit-app
```

## Step 6: Configure Streamlit (Optional but Recommended)

### 6.1 Create Streamlit Config
```bash
mkdir -p ~/.streamlit
nano ~/.streamlit/config.toml
```

Add:
```toml
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## Step 7: Access Your Application

### 7.1 Get Your Public IP
```bash
curl http://169.254.169.254/latest/meta-data/public-ipv4
```

### 7.2 Access in Browser
Open your browser and go to:
```
http://YOUR_PUBLIC_IP:8501
```

## Step 8: Set Up Domain Name (Optional)

### 8.1 Using Route 53
1. Register a domain or use existing one
2. Create A record pointing to your EC2 instance IP
3. Update security group to allow port 80/443

### 8.2 Using Nginx Reverse Proxy (Recommended)

**Install Nginx:**
```bash
# Amazon Linux
sudo yum install nginx -y

# Ubuntu
sudo apt install nginx -y
```

**Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/streamlit
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

**Enable and Start:**
```bash
# Ubuntu
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Amazon Linux
sudo systemctl start nginx
sudo systemctl enable nginx
```

## Step 9: Set Up SSL with Let's Encrypt (Optional but Recommended)

### 9.1 Install Certbot
```bash
# Ubuntu
sudo apt install certbot python3-certbot-nginx -y

# Amazon Linux
sudo yum install certbot python3-certbot-nginx -y
```

### 9.2 Get SSL Certificate
```bash
sudo certbot --nginx -d your-domain.com
```

Follow the prompts. Certbot will automatically configure Nginx.

## Step 10: Monitoring and Maintenance

### 10.1 View Logs
```bash
# Application logs
sudo journalctl -u streamlit-app -f

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### 10.2 Restart Application
```bash
sudo systemctl restart streamlit-app
```

### 10.3 Update Application
```bash
cd ~/YOUR_PROJECT_DIRECTORY
git pull  # if using git
# or upload new files via SCP
source venv/bin/activate
pip install -r requirements.txt  # if dependencies changed
sudo systemctl restart streamlit-app
```

## Step 11: Firewall Configuration (UFW for Ubuntu)

If using Ubuntu, configure UFW:
```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8501/tcp
sudo ufw enable
```

## Troubleshooting

### Application Not Accessible
1. Check security group rules
2. Check if service is running: `sudo systemctl status streamlit-app`
3. Check logs: `sudo journalctl -u streamlit-app -n 50`
4. Verify port is listening: `sudo netstat -tulpn | grep 8501`

### Permission Issues
```bash
sudo chown -R ec2-user:ec2-user ~/YOUR_PROJECT_DIRECTORY
chmod +x ~/YOUR_PROJECT_DIRECTORY/app.py
```

### Port Already in Use
```bash
sudo lsof -i :8501
sudo kill -9 <PID>
```

### Update Models
If you need to update model files:
```bash
# Upload new models via SCP
scp -i your-key-pair.pem models/*.pth ec2-user@YOUR_IP:~/YOUR_PROJECT_DIRECTORY/models/
scp -i your-key-pair.pem models/*.pkl ec2-user@YOUR_IP:~/YOUR_PROJECT_DIRECTORY/models/
```

## Cost Optimization Tips

1. **Use Reserved Instances** for long-term deployments
2. **Stop instance** when not in use (data persists on EBS)
3. **Use t2.micro/t3.micro** for development/testing
4. **Enable CloudWatch alarms** to monitor costs
5. **Use S3** for storing large model files instead of EBS

## Security Best Practices

1. **Restrict SSH access** to your IP only
2. **Use key pairs** instead of passwords
3. **Keep system updated**: `sudo yum update` or `sudo apt update`
4. **Use HTTPS** with Let's Encrypt
5. **Regular backups** of your application and data
6. **Monitor logs** for suspicious activity
7. **Use IAM roles** instead of storing AWS credentials

## Quick Start Script

Save this as `deploy.sh` and run it on your EC2 instance:

```bash
#!/bin/bash
set -e

echo "Setting up Streamlit app..."

# Update system
sudo yum update -y || sudo apt update && sudo apt upgrade -y

# Install Python
sudo yum install python3 python3-pip git -y || sudo apt install python3 python3-pip python3-venv git -y

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install streamlit pandas numpy plotly requests beautifulsoup4 joblib torch pyyaml lightgbm

echo "Setup complete! Now:"
echo "1. Upload your project files"
echo "2. Configure the systemd service"
echo "3. Start the service"
```

Make it executable:
```bash
chmod +x deploy.sh
./deploy.sh
```

## Support

For issues:
1. Check application logs
2. Check system logs
3. Verify security group settings
4. Check AWS service health dashboard

---

**Note:** Replace all placeholders (YOUR_PUBLIC_IP, YOUR_PROJECT_DIRECTORY, etc.) with your actual values.

