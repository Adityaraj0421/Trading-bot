# ============================================================
# Dockerfile - Build and run the dashboard in a container
# ============================================================
# Usage:
#   docker build -t crypto-dashboard ./dashboard
#   docker run -p 3000:3000 crypto-dashboard

FROM node:20-alpine

WORKDIR /app

# Copy package files first (Docker caches this layer)
COPY package*.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Build the Next.js production bundle
RUN npm run build

# Expose port 3000 (Next.js default)
EXPOSE 3000

# Start the production server
CMD ["npm", "start"]
