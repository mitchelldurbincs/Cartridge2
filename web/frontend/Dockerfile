# Frontend Service Dockerfile
# Multi-stage build: Node.js build + nginx serve

# ============================================
# Stage 1: Build Frontend (Node.js)
# ============================================
FROM node:20-bookworm-slim AS builder

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source
COPY . ./

# Build the frontend
RUN npm run build

# ============================================
# Stage 2: Serve with nginx (non-root)
# ============================================
FROM nginx:1.27-alpine

# Create non-root user and set up permissions
RUN adduser -D -u 1000 cartridge && \
    chown -R cartridge:cartridge /usr/share/nginx/html && \
    chown -R cartridge:cartridge /var/cache/nginx && \
    chown -R cartridge:cartridge /var/log/nginx && \
    chown cartridge:cartridge /etc/nginx/conf.d

# Copy built frontend to nginx html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8080

# Healthcheck to verify nginx is serving content
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/ || exit 1

# Run as non-root user
USER cartridge

CMD ["nginx", "-g", "daemon off;"]
