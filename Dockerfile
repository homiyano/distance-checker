FROM node:20-alpine AS build
WORKDIR /app/web
COPY web/package.json web/package-lock.json* ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 80
