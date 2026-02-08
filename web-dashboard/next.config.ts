import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker
  output: 'standalone',

  // Disable static optimization for dynamic data
  experimental: {
    // Enable if needed for better performance
  }
};

export default nextConfig;
