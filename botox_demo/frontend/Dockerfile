# Multi-stage build: compile the Vite bundle, then serve it with nginx and proxy /api to backend.

# ---- build stage ----
FROM node:20-alpine AS build
WORKDIR /app

# Install deps first (layer-cached unless the manifests change). `npm ci` is a reproducible install
# from the committed lockfile and fails hard if the lockfile is missing or out of sync, which is what
# a production image wants: no silent fallback to a non-reproducible `npm install` that could pull
# newer, unvetted dependency versions. The lockfile is committed, so require it (no `*` glob).
COPY package.json package-lock.json ./
RUN npm ci

# Build the static bundle.
COPY tsconfig.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build

# ---- serve stage ----
FROM nginx:1.27-alpine AS serve

# Our server config (SPA + /api proxy to the backend service) + the shared security-headers snippet
# it includes in each location.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY security-headers.conf /etc/nginx/security-headers.conf

# The built assets.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
# Target 127.0.0.1, not "localhost": in the container localhost resolves to ::1 (IPv6) first, but
# nginx listens on 0.0.0.0:80 (IPv4), so a localhost check hits a refused IPv6 connection.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1/ >/dev/null 2>&1 || exit 1

CMD ["nginx", "-g", "daemon off;"]
