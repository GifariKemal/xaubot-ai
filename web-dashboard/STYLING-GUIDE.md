# XAUBot AI Dashboard - Styling Guide

## Overview

The dashboard uses **shadcn/ui** components with **Tailwind CSS** and a custom dark theme inspired by nof1.ai and SURGE-AI-Trading.

## Color System

### Theme Colors
```typescript
// Background & Surface
background: #0a0a0f (HSL: 222 47% 6%)
surface: #121218 (HSL: 222 25% 7%)
surface-light: #1a1a24 (HSL: 222 20% 10%)
surface-hover: #22222e (HSL: 222 18% 14%)

// Primary & Accent
primary: #6366f1 (Indigo)
primary-dark: #4f46e5
accent: #8b5cf6 (Purple)

// Semantic Colors
success: #22c55e (Green)
warning: #f59e0b (Orange)
danger: #ef4444 (Red)
info: #3b82f6 (Blue)

// Each semantic color has a background variant with 12.5% opacity
success-bg: #22c55e20
warning-bg: #f59e0b20
danger-bg: #ef444420
info-bg: #3b82f620
```

### Border & Text
```typescript
border: #2a2a3a
border-light: #3a3a4a
foreground: #ffffff
muted-foreground: #a1a1aa
```

## Component Styling

### Cards
```tsx
// Glass effect card (recommended for dashboard)
<Card className="glass">
  <CardHeader>...</CardHeader>
  <CardContent>...</CardContent>
</Card>

// Custom card utilities
.glass → bg-surface/80 + backdrop-blur
.card-custom → bg-surface + rounded-xl + border
.card-hover → card-custom + hover effect
```

### Badges
```tsx
// Available badge variants
<Badge variant="default">Primary</Badge>
<Badge variant="success">Success</Badge>
<Badge variant="warning">Warning</Badge>
<Badge variant="danger">Danger</Badge>
<Badge variant="info">Info</Badge>
<Badge variant="outline">Outline</Badge>
```

### Buttons
```tsx
// Utility classes for buttons
className="btn-primary"  → Primary button
className="btn-success"  → Success button
className="btn-danger"   → Danger button
className="btn-outline"  → Outline button
```

## Utility Classes

### Text & Numbers
```css
.font-number → font-mono + tabular-nums (for prices, numbers)
.text-gradient → gradient from primary to accent

.price-up → text-success
.price-down → text-danger
.price-neutral → text-muted-foreground
```

### Animations
```css
.animate-pulse-slow → 3s pulse
.animate-fade-in → fade in effect
.animate-slide-up → slide up effect
.animate-shimmer → shimmer loading effect
.skeleton → loading skeleton with shimmer
```

### Live Indicators
```tsx
// Adds a pulsing dot indicator
<div className="pulse-live">LIVE</div>
```

## Utility Functions

### Formatting
```typescript
import {
  formatUSD,        // → $1,234.56
  formatGoldPrice,  // → 2345.67
  formatPercent,    // → +2.45%
  formatCompact,    // → 1.2M, 3.4K
  formatTime,       // → 14:23:45
  formatDate,       // → Jan 17, 2026
  formatDateTime,   // → Jan 17, 2026 14:23:45
  formatDateTimeWIB // → 17 Jan 2026 14:23:45 WIB
} from '@/lib/utils';
```

### Color Helpers
```typescript
import {
  getValueColor,        // → Returns color class based on +/-
  getValueBgColor,      // → Returns bg color class based on +/-
  getSignalColor,       // → Returns color for BUY/SELL/HOLD
  getSignalBadgeColor,  // → Returns badge variant for signals
  getConfidenceColor,   // → Returns color based on confidence %
  getConfidenceLevel    // → Returns "Very High", "High", etc.
} from '@/lib/utils';
```

### Other Utilities
```typescript
import {
  cn,              // Merge Tailwind classes
  calcProgress,    // Calculate progress % (capped at 100)
  debounce,        // Debounce function
  generateId,      // Generate unique ID
  sleep           // Async sleep
} from '@/lib/utils';
```

## Usage Examples

### Price Display
```tsx
import { formatGoldPrice, getValueColor } from '@/lib/utils';

<span className={cn(
  "text-3xl font-bold font-number",
  getValueColor(priceChange)
)}>
  ${formatGoldPrice(price)}
</span>
```

### Signal Badge
```tsx
import { getSignalBadgeColor } from '@/lib/utils';

<Badge variant={getSignalBadgeColor(signal)}>
  {signal}
</Badge>
```

### Confidence Display
```tsx
import { getConfidenceColor, getConfidenceLevel } from '@/lib/utils';

const confidencePercent = confidence * 100;

<span className={cn(
  "font-semibold",
  getConfidenceColor(confidencePercent)
)}>
  {confidencePercent.toFixed(0)}% - {getConfidenceLevel(confidencePercent)}
</span>
```

### Profit/Loss Display
```tsx
import { formatUSD, getValueColor } from '@/lib/utils';

<span className={cn(
  "font-bold font-number",
  getValueColor(profit)
)}>
  {profit >= 0 ? '+' : ''}{formatUSD(profit)}
</span>
```

## Typography

### Fonts
- **Sans:** Inter, system-ui, sans-serif
- **Mono:** JetBrains Mono, Fira Code, monospace

### Font Classes
```tsx
<span className="font-sans">Regular text</span>
<span className="font-mono">Code or numbers</span>
<span className="font-number">Numbers (tabular-nums)</span>
```

## Responsive Design

The dashboard is optimized for desktop but responsive:
```tsx
<div className="hidden sm:flex">Desktop only</div>
<div className="sm:hidden">Mobile only</div>
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
  Responsive grid
</div>
```

## Adding New shadcn/ui Components

1. Check available components:
```bash
npx shadcn@latest view @shadcn
```

2. Add a component:
```bash
npx shadcn@latest add button
npx shadcn@latest add tooltip
npx shadcn@latest add dialog
```

3. Components will be added to `src/components/ui/`

## Best Practices

1. **Always use utility functions** for formatting numbers, dates, and colors
2. **Use the `cn()` helper** to merge Tailwind classes
3. **Apply `font-number`** to all numeric displays for consistent monospace formatting
4. **Use semantic colors** (success, warning, danger, info) instead of raw colors
5. **Apply `glass` effect** to cards for depth and consistency
6. **Use uppercase + tracking-wider** for card titles: `className="uppercase tracking-wider"`
7. **Add proper spacing** with `space-y-*` or `gap-*` utilities
8. **Keep contrast in mind** - use `text-muted-foreground` for secondary text

## Example Card Component

```tsx
"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { TrendingUp } from "lucide-react";
import { cn, formatUSD, getValueColor } from "@/lib/utils";

interface ExampleCardProps {
  title: string;
  value: number;
  change: number;
  status: "active" | "inactive";
}

export function ExampleCard({ title, value, change, status }: ExampleCardProps) {
  return (
    <Card className="glass">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2 uppercase tracking-wider">
          <TrendingUp className="h-4 w-4" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-2xl font-bold font-number">
          {formatUSD(value)}
        </div>
        <div className="flex items-center justify-between">
          <span className={cn(
            "text-sm font-medium font-number",
            getValueColor(change)
          )}>
            {change >= 0 ? '+' : ''}{change.toFixed(2)}%
          </span>
          <Badge variant={status === 'active' ? 'success' : 'danger'}>
            {status}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
```

## Migration Checklist

When updating existing components to the new styling:

- [ ] Replace hardcoded colors with theme colors (text-green-500 → text-success)
- [ ] Add `glass` class to cards
- [ ] Use utility formatting functions instead of manual formatting
- [ ] Apply `font-number` to numeric displays
- [ ] Use uppercase + tracking-wider for titles
- [ ] Replace manual color logic with utility functions (getValueColor, etc.)
- [ ] Update Badge variants to semantic ones (success, warning, danger, info)
- [ ] Add proper spacing with space-y or gap utilities
- [ ] Ensure proper use of `cn()` for class merging

## Resources

- shadcn/ui docs: https://ui.shadcn.com
- Tailwind CSS docs: https://tailwindcss.com
- Lucide Icons: https://lucide.dev

---

Last updated: Feb 6, 2026
