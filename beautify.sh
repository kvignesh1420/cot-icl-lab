#!/bin/bash

echo "🚀 Fixing import order with isort..."
isort --float-to-top .

echo "🚀 Running Ruff Formatting..."
ruff format .

echo "🚀 Running Ruff Linting with Fixes..."
ruff check --fix .

echo "✅ Ruff formatting and linting completed!"
