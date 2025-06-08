#!/bin/bash

# Macha  Deployment Script
# Deploys the systemd service for automatic startup

set -e

SERVICE_NAME="macha"
SERVICE_FILE="macha.service"
SYSTEMD_DIR="/etc/systemd/system"
USER="payload"
MACHA_DIR="/home/payload/macha"

echo "Deploying Macha  Service..."
echo "=================================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (use sudo)"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "Error: Service file '$SERVICE_FILE' not found"
    echo "Make sure you're running this from the macha directory"
    exit 1
fi

# Check if macha directory exists
if [ ! -d "$MACHA_DIR" ]; then
    echo "Error: Macha directory '$MACHA_DIR' not found"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$MACHA_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at '$MACHA_DIR/.venv'"
    echo "Run 'uv sync' first"
    exit 1
fi

# Check if user exists
if ! id "$USER" &>/dev/null; then
    echo "Error: User '$USER' does not exist"
    exit 1
fi

# Stop service if it's already running
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "Stopping existing $SERVICE_NAME service..."
    systemctl stop "$SERVICE_NAME"
fi

# Copy service file to systemd directory
echo "Installing service file..."
cp "$SERVICE_FILE" "$SYSTEMD_DIR/"
chmod 644 "$SYSTEMD_DIR/$SERVICE_FILE"

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable service for automatic startup
echo "Enabling $SERVICE_NAME service..."
systemctl enable "$SERVICE_NAME"

# Start the service
echo "Starting $SERVICE_NAME service..."
systemctl start "$SERVICE_NAME"

# Wait a moment and check status
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✓ Service started successfully"
else
    echo "✗ Service failed to start"
    echo "Check logs with: journalctl -u $SERVICE_NAME -f"
    exit 1
fi

# Show service status
echo ""
echo "Service Status:"
systemctl status "$SERVICE_NAME" --no-pager -l

echo ""
echo "Deployment completed successfully!"
echo ""
echo "Useful commands:"
echo "  View logs:        journalctl -u $SERVICE_NAME -f"
echo "  Service status:   systemctl status $SERVICE_NAME"
echo "  Stop service:     sudo systemctl stop $SERVICE_NAME"
echo "  Start service:    sudo systemctl start $SERVICE_NAME"
echo "  Restart service:  sudo systemctl restart $SERVICE_NAME"
echo "  Disable service:  sudo systemctl disable $SERVICE_NAME"
echo ""
echo "The service will now start automatically on boot."
