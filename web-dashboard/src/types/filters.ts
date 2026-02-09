export interface FilterConfig {
  enabled: boolean;
  name: string;
  description: string;
}

export interface FilterConfigData {
  filters: Record<string, FilterConfig>;
  metadata: {
    updated_at: string;
    version: string;
  };
}

export interface FilterUpdate {
  [key: string]: boolean;
}
