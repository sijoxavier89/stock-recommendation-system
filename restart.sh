#!/bin/bash

echo "Restarting all services..."
echo ""

./stop.sh

echo ""
echo "Waiting 2 seconds before restart..."
sleep 2

./start.sh
