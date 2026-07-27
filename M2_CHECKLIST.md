Module: reports/ItemSalesReport/compare.php
URL: https://mark.solucioneshys.net/reports/ItemSalesReport/compare.php

Bug 1 — Charts render empty (axes/labels show, no plotted data)

In the "Department Growth — Year vs Year" cards (Outdoor, Produce, etc.), the totals and % change display correctly (e.g. "2026: $309,876.93 · 2025: $397,175.17, -22.0%"), so the underlying query data is correct. But the chart area itself shows only the Y-axis scale, X-axis, and season-band background colors — no line/bar series is drawn.

Likely causes to check in compare.php (and any shared chart JS it includes, e.g. in mark.js or an inline <script> block):

The chart-drawing JS may be running before the DOM/canvas element exists (script placed above the canvas, or missing DOMContentLoaded/deferred load).
The series data array passed to the chart function may be empty, malformed, or using the wrong key name (mismatch between what PHP echoes into JS — e.g. json_encode($weeklySeriesY1) — and what the chart-drawing function expects).
If this uses a <canvas> with manual JS (not a library like Chart.js), check that ctx.beginPath()/moveTo()/lineTo()/stroke() calls are actually being reached, and that coordinate scaling isn't producing NaN (e.g. dividing by a max value of 0 when a week has no data).
Check the browser console for JS errors on this page — a thrown error partway through chart setup would abort rendering silently for all charts on the page.
Confirm the two "Outdoor" and "Produce" charts use independently-scoped variables/IDs — if canvas IDs collide or a shared array is being overwritten before both charts render, only one (or neither) would draw.

Ask: Inspect compare.php's chart-rendering code end-to-end, find why the series data isn't being plotted, and fix it so both the 2026 and 2025 lines/bars actually render on the weekly chart, matching the totals already shown above each chart.

Bug 2 — X-axis shows week numbers (W1, W2...) instead of dates

The weekly chart's X-axis is labeled with ISO week numbers (W1, W2, W3...). Staff find week numbers hard to read — they think in calendar dates, not week-of-year numbers.

Ask: Change the X-axis labels to calendar dates instead of week numbers. Specifically:

Use the week_start_date (already available from the underlying query, same field used elsewhere in this module like trends.php) instead of the ISO week number for axis labels.
Format as a short, readable date, e.g. Mar 3 or Mar 3, 2026 (per project convention, F j, Y style, shortened for axis space — something like M j).
Keep the season-band legend (Winter/Spring/Summer/Harvest/Holiday) working correctly against the new date-based axis — the color bands should still align to the correct weeks.
Apply this consistently to both charts in the Department Growth comparison and any other week-numbered chart on this page (e.g. the "Item & Category Growth Explorer" section further down, if it also uses week numbers).
