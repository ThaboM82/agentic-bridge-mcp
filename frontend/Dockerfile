FROM node:20-alpine

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json* ./
RUN npm install

# Copy source files
COPY . .

# Build or run development server
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]