#!/bin/bash
# chmod +x save_db_update_files.sh
# ./save_db_update_files.sh
# Script to update auctionator.db, restore with git, pull changes, and restart docker-compose

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Move auctionator.db to asd.db
echo "Moving 'auctionator.db' to 'asd.db'..."
mv auctionator.db asd.db

# Step 2: Restore auctionator.db using git
echo "Restoring 'auctionator.db' from git..."
git restore auctionator.db

# Step 3: Pull the latest changes from git
echo "Pulling latest changes from git..."
git pull

# Step 4: Move asd.db back to auctionator.db
echo "Moving 'asd.db' back to 'auctionator.db'..."
mv asd.db auctionator.db

# Step 5: Restart docker-compose in detached mode
echo "Restarting docker-compose in detached mode..."
docker compose down
docker compose up -d

echo "Script completed successfully!"
