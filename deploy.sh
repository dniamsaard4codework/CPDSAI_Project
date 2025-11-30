#!/bin/bash
# Quick deployment script for AWS EC2
# Run this script on your EC2 instance after uploading your project

set -e

echo "🚀 Starting deployment of Water Level Monitoring App..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Exiting."
    exit 1
fi

echo -e "${BLUE}Detected OS: $OS${NC}"

# Update system
echo -e "${BLUE}Updating system packages...${NC}"
if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y python3 python3-pip python3-venv git nginx
elif [ "$OS" = "amzn" ] || [ "$OS" = "amazon" ]; then
    sudo yum update -y
    sudo yum install -y python3 python3-pip git
else
    echo "Unsupported OS. Please install dependencies manually."
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Installing basic packages..."
    pip install streamlit pandas numpy plotly requests beautifulsoup4 joblib pyyaml
fi

# Create systemd service
echo -e "${BLUE}Creating systemd service...${NC}"
PROJECT_DIR=$(pwd)
USER=$(whoami)

sudo tee /etc/systemd/system/streamlit-app.service > /dev/null <<EOF
[Unit]
Description=Streamlit Water Level Monitoring App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create Streamlit config directory
echo -e "${BLUE}Creating Streamlit configuration...${NC}"
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml <<EOF
[server]
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
headless = true

[browser]
gatherUsageStats = false
serverAddress = "localhost"
EOF

# Reload systemd and enable service
echo -e "${BLUE}Enabling and starting service...${NC}"
sudo systemctl daemon-reload
sudo systemctl enable streamlit-app
sudo systemctl start streamlit-app

# Wait a moment for service to start
sleep 3

# Check service status
if sudo systemctl is-active --quiet streamlit-app; then
    echo -e "${GREEN}✅ Service started successfully!${NC}"
    echo -e "${GREEN}Your app should be accessible at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501${NC}"
else
    echo -e "${BLUE}⚠️  Service may have issues. Check status with: sudo systemctl status streamlit-app${NC}"
    echo -e "${BLUE}View logs with: sudo journalctl -u streamlit-app -f${NC}"
fi

echo -e "${GREEN}🎉 Deployment complete!${NC}"
echo ""
echo "Useful commands:"
echo "  - View logs: sudo journalctl -u streamlit-app -f"
echo "  - Restart: sudo systemctl restart streamlit-app"
echo "  - Status: sudo systemctl status streamlit-app"
echo "  - Stop: sudo systemctl stop streamlit-app"

