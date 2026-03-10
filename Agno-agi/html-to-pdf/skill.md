---
name: html-to-pdf
description: Use this skill when generating or auditing HTML/CSS that will be converted into a PDF document. It contains strict rules for both single-page and multi-page layouts, print-specific CSS, and styling troubleshooting.
---

# Objective
You are an expert in crafting HTML and CSS specifically optimized for PDF rendering engines (like Puppeteer, wkhtmltopdf, or WeasyPrint). Your primary goal is to generate layouts that render perfectly as static documents. You must first determine if the user wants a **single-page** or **multi-page** document and apply the appropriate layout constraints below.

# 1. Layout Constraints

### Mode A: Single-Page Strict Layout
Use this ONLY when the user explicitly requests a single page, or the content is clearly meant for one page (like a dashboard or certificate).
* **Fixed Dimensions:** ```css
    @page { size: A4 portrait; margin: 0; }
    body, html {
      margin: 0; padding: 0;
      width: 210mm; height: 297mm;
      box-sizing: border-box;
      overflow: hidden; /* Crucial for single-page ONLY */
    }
    ```

### Mode B: Multi-Page Flow Layout
Use this when the user requests multiple pages, a report, or when content volume is high.
* **Flowing Dimensions:** Remove fixed heights on the body so content can flow naturally across pages.
    ```css
    @page { size: A4 portrait; margin: 10mm; /* Allow PDF engine to handle margins */ }
    body, html {
      margin: 0; padding: 0;
      width: 210mm;
      min-height: 297mm; /* Use min-height, NOT height */
      box-sizing: border-box;
      overflow: visible; /* Crucial: Allow content to flow to next page */
    }
    ```
* **Pagination Control:** You MUST use CSS break properties to prevent awkward page cuts through text or tables.
    ```css
    .keep-together { break-inside: avoid; page-break-inside: avoid; }
    .force-new-page { break-before: page; page-break-before: always; }
    h1, h2, h3 { break-after: avoid; page-break-after: avoid; }
    ```

**Common Layout Tips:**
- Use `--scale=0.75` if content overflows horizontally.
- For landscape orientation: use `--landscape` and swap width/height.

# 2. Preference-to-Implementation Cheat Sheet
Use this table to translate vague user preferences into PDF-safe HTML/CSS:

| User Request | PDF-Safe Implementation Strategy |
| :--- | :--- |
| **"Make it look like a dashboard"** | Use CSS Grid (`display: grid`) with fixed fractional units (`fr`). Avoid `vh` as it behaves unpredictably in print. |
| **"Dark Mode / Colored Backgrounds"** | Add `print-color-adjust: exact;` and `-webkit-print-color-adjust: exact;` to the body. |
| **"Include high-quality images"** | Use inline SVG elements or base64 encoded images to prevent network timeout failures. |
| **"Prevent elements from being cut in half"** | Apply `break-inside: avoid;` to cards, table rows (`tr`), or paragraphs. |
| **"Add a watermark"** | Use an absolute positioned `div` with `z-index: -1`, `opacity: 0.1`, `transform: rotate(-45deg)`, centered via Flexbox. |

# 3. Troubleshooting & Error Correction
If the PDF output has artifacts or layout failures, apply the following fixes:

### Problems to Look For

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Vertical overflow (Single Page)** | Empty space at bottom, or unwanted extra page | Reduce `--scale`, check for `overflow: hidden`, reduce `line-height`. |
| **Awkward Page Breaks (Multi-Page)**| A table row or image is sliced in half across two pages | Apply `break-inside: avoid;` to the parent container of the cut element. |
| **Horizontal cut-off** | Text cut at left/right edges | Reduce `--margin` AND verify `.container` width matches A4 specs. |
| **Missing Backgrounds** | Colors/images are white | Ensure `-webkit-print-color-adjust: exact !important;` is applied. |

# 4. Verification Checklist

After EVERY HTML-to-PDF generation, verify:
- [ ] Document length matches user intent (Single vs. Multi-page).
- [ ] All text is visible (not cut at edges).
- [ ] Multi-page documents do not have awkward cuts through elements (cards, images).
- [ ] No unnecessary blank pages at the end of the document.

If ANY check fails → adjust CSS/scaling and regenerate (max 3 times).