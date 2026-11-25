export type Tier1ToolName =
  | 'read_active_page'
  | 'web_search'
  | 'fetch_and_read_url'
  | 'list_open_tabs'
  | 'read_tab'
  | 'js_calc';

export type PageExtractMode = 'auto' | 'article' | 'selection' | 'interactive';

export interface ParsedToolCall {
  name: Tier1ToolName;
  arguments: Record<string, unknown>;
  raw?: string;
}

export interface ToolExecutionResult {
  ok: boolean;
  toolName: Tier1ToolName;
  data?: unknown;
  formatted?: string;
  error?: string;
  needsPermission?: boolean;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
}

export interface PendingToolDetection {
  isOpen: boolean;
  toolName: string | null;
  label: string;
}

export interface ToolParameterSchema {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object';
  description: string;
  enum?: string[];
  items?: { type: string };
  minimum?: number;
  maximum?: number;
}

export interface ToolDefinition {
  name: Tier1ToolName;
  description: string;
  parameters: {
    type: 'object';
    properties: Record<string, ToolParameterSchema>;
    required?: string[];
  };
  requiresHostPermission?: boolean;
}

export interface ExtractedPageContent {
  mode: 'article' | 'selection' | 'youtube' | 'pdf' | 'interactive';
  title: string;
  url: string;
  content: string;
  contentHash: string;
  charCount: number;
  truncated: boolean;
  error?: string;
  youtubeCaptionUrl?: string;
  youtubeLanguageCode?: string;
}

export interface SearchResultItem {
  title: string;
  url: string;
  snippet: string;
  domain: string;
}

export interface WebSearchResult {
  ok: boolean;
  query: string;
  results: SearchResultItem[];
  fromHiddenTab?: boolean;
  needsPermission?: boolean;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
  error?: string;
}

export interface FetchedSourceResult {
  url: string;
  title: string;
  markdown: string;
  excerpt: string;
  charCount: number;
  fromHiddenTab: boolean;
  error?: string;
  blockedByCaptcha?: boolean;
  manualUrl?: string;
}

export interface OpenTabInfo {
  id: number;
  title: string;
  url: string;
  active: boolean;
  windowId: number;
}

export interface HiddenTabHtmlResponse {
  html: string;
  title?: string;
  url?: string;
}

export interface JsCalcInput {
  expression?: string;
  mode?: 'expr' | 'unit' | 'date' | 'json' | 'csv';
  value?: unknown;
  from?: string;
  to?: string;
  dateA?: string;
  dateB?: string;
  days?: number;
  data?: string;
  path?: string;
}

export interface JsCalcResult {
  ok: boolean;
  result?: unknown;
  formatted: string;
  error?: string;
}
