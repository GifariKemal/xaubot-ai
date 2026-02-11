# XAUBot Pro V3 - Troubleshooting Guide

## Common Errors & Solutions

### Error: "cannot load 'XAUBot_Pro_V3'"
**Cause:** File corrupted or compilation issue
**Solution:**
1. Re-compile EA in MetaEditor
2. Or use the .ex5 file that was already compiled

### Error: "DLL imports not allowed"
**Solution:**
1. EA Settings → Tab "Common" → ☑ Allow DLL imports
2. Tools → Options → Expert Advisors → ☑ Allow DLL imports

### Error: "AutoTrading disabled by client"
**Solution:**
1. Click "Algo Trading" button on toolbar (make it GREEN)
2. Or press Alt+E

### Error: "Invalid stops" or "Invalid SL/TP"
**Cause:** Broker restrictions on stop levels
**Solution:**
1. Check symbol specifications: Right-click chart → Specification
2. Look for "Stops level" - if >0, EA will auto-adjust

### Error: Panel tidak muncul
**Solution:**
1. Check input: ShowPanel = true
2. Check PanelOffsetX/Y (default 380, 10)
3. Try different PanelCorner (CORNER_LEFT_UPPER → CORNER_RIGHT_UPPER)
4. Restart EA (remove from chart, attach again)

### Error: "INIT_FAILED"
**Check Tab Experts for specific reason:**
- "Failed to create indicators" → Wrong timeframe or symbol
- "Failed to set symbol" → Symbol name incorrect (must be XAUUSD)
- Handle errors → Indicator loading issue

### Error: No trades after 24 hours
**THIS IS NORMAL!**
- EA rejects 90%+ of signals
- Average 8-20 trades per MONTH (not per day!)
- Check log file for filter rejections
- Verify quality score is being calculated (check panel)

## Diagnostic Commands

### Check if file exists:
```bash
ls -lh "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/010E047102812FC0C18890992854220E/MQL5/Experts/XAUBot_Pro_V3.ex5"
```

### Check log file exists:
```bash
ls -lh "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/010E047102812FC0C18890992854220E/MQL5/Files/XAUBot_V3_*.log"
```

### Read recent log entries:
```bash
tail -n 50 "C:/Users/Administrator/AppData/Roaming/MetaQuotes/Terminal/010E047102812FC0C18890992854220E/MQL5/Files/XAUBot_V3_2026-02-10.log"
```

## Files Location

### EA Location:
```
C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\
010E047102812FC0C18890992854220E\MQL5\Experts\
├── XAUBot_Pro_V3.ex5 (67 KB) - Compiled EA
└── XAUBot_Pro_V3.mq5 (44 KB) - Source code
```

### Log Location:
```
C:\Users\Administrator\AppData\Roaming\MetaQuotes\Terminal\
010E047102812FC0C18890992854220E\MQL5\Files\
└── XAUBot_V3_YYYY-MM-DD.log - Daily log file
```

## Quick Test

1. Attach EA to XAUUSD M15 chart
2. Wait 1 minute
3. Check for panel display
4. Check tab "Experts" for initialization message
5. Check Files folder for log file creation

If all 3 checks pass → EA is working! ✅

## Contact Info

If EA still not working after all troubleshooting:
1. Screenshot tab "Experts" (full error message)
2. Screenshot chart (show emoticon status)
3. Share log file content (first 50 lines)
