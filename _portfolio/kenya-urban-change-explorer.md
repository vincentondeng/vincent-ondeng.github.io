---
title: "Kenya Urban Change STAC Explorer"
excerpt: "Multi-temporal NDVI and built-up composites for Nairobi, Mombasa,
Kisumu, and Nakuru — tracking a decade of urban growth using Sentinel-2
imagery and a reproducible Jupyter notebook."
collection: portfolio
header:
  teaser: /images/kenya-urban-change-thumb.png
---

<p>
Kenya's cities are growing fast. Between 2015 and 2024, Nairobi, Mombasa,
Kisumu, and Nakuru have each undergone significant urban expansion —
green space lost, built-up land gained, and city boundaries pushed outward.
This project makes that change visible, measurable, and reproducible.
</p>

<p>
Built on top of the
<a href="/portfolio/stac-vegetation-pipeline/">STAC Vegetation Index Pipeline</a>,
this explorer generates annual median composites from Sentinel-2 imagery for
each city, computes multi-temporal NDVI to track vegetation loss, and derives
built-up indices (NDBI, BUI) to quantify urban expansion — all from a single
parametrized notebook you can run on any city with a STAC-compliant data source.
</p>

<p>
Outputs include per-city change statistics, Cloud Optimized GeoTIFF exports for
each year, and a methodology note documenting the full analytical approach.
</p>

<hr/>

<h3>Pipeline</h3>

<svg width="100%" viewBox="0 0 680 580" role="img" xmlns="http://www.w3.org/2000/svg">
  <title>Kenya Urban Change STAC Explorer Pipeline</title>
  <desc>
    Pipeline from city selection and STAC query through multi-temporal NDVI
    and built-up composite generation to reproducible notebook output.
  </desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
    <style>
      .box-gray   { fill: #F1EFE8; stroke: #888780; }
      .box-teal   { fill: #E1F5EE; stroke: #0F6E56; }
      .box-green  { fill: #EAF3DE; stroke: #3B6D11; }
      .box-purple { fill: #EEEDFE; stroke: #534AB7; }
      .lbl-gray   { fill: #444441; font-family: sans-serif; }
      .lbl-teal   { fill: #085041; font-family: sans-serif; }
      .lbl-green  { fill: #27500A; font-family: sans-serif; }
      .lbl-purple { fill: #3C3489; font-family: sans-serif; }
      .lbl-sub    { fill: #5F5E5A; font-family: sans-serif; }
      .lbl-step   { fill: #B4B2A9; font-family: sans-serif; }
      .arr-line   { stroke: #888780; stroke-width: 1.5; fill: none;
                    marker-end: url(#arrow); }
    </style>
  </defs>

  <!-- Step 1: City selection -->
  <rect class="box-gray" x="220" y="20" width="240" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-gray" x="340" y="42" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">City selection</text>
  <text class="lbl-sub" x="340" y="60" font-size="12"
        text-anchor="middle" dominant-baseline="central">Nairobi · Mombasa · Kisumu · Nakuru</text>
  <text class="lbl-step" x="650" y="52" font-size="11" text-anchor="end">01</text>

  <line class="arr-line" x1="340" y1="76" x2="340" y2="108"/>

  <!-- Step 2: STAC query -->
  <rect class="box-teal" x="160" y="108" width="360" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="130" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">STAC query — Sentinel-2 (2015–2024)</text>
  <text class="lbl-sub" x="340" y="148" font-size="12"
        text-anchor="middle" dominant-baseline="central">Reuses STAC Vegetation Index Pipeline module</text>
  <text class="lbl-step" x="650" y="140" font-size="11" text-anchor="end">02</text>

  <line class="arr-line" x1="340" y1="164" x2="340" y2="196"/>

  <!-- Step 3: Annual composites -->
  <rect class="box-teal" x="160" y="196" width="360" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="218" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Annual composite generation</text>
  <text class="lbl-sub" x="340" y="236" font-size="12"
        text-anchor="middle" dominant-baseline="central">Median composites per year · cloud masking</text>
  <text class="lbl-step" x="650" y="228" font-size="11" text-anchor="end">03</text>

  <!-- Split arrows -->
  <line class="arr-line" x1="280" y1="252" x2="180" y2="284"/>
  <line class="arr-line" x1="400" y1="252" x2="500" y2="284"/>

  <!-- Step 4a: NDVI -->
  <rect class="box-green" x="60" y="284" width="240" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-green" x="180" y="306" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Multi-temporal NDVI</text>
  <text class="lbl-sub" x="180" y="324" font-size="12"
        text-anchor="middle" dominant-baseline="central">Vegetation change 2015 → 2024</text>
  <text class="lbl-step" x="50" y="316" font-size="11" text-anchor="start">04a</text>

  <!-- Step 4b: Built-up -->
  <rect class="box-purple" x="380" y="284" width="240" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-purple" x="500" y="306" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Built-up composite</text>
  <text class="lbl-sub" x="500" y="324" font-size="12"
        text-anchor="middle" dominant-baseline="central">NDBI / BUI · urban expansion index</text>
  <text class="lbl-step" x="650" y="316" font-size="11" text-anchor="end">04b</text>

  <!-- Merge arrows -->
  <line class="arr-line" x1="180" y1="340" x2="280" y2="372"/>
  <line class="arr-line" x1="500" y1="340" x2="400" y2="372"/>

  <!-- Step 5: Change detection -->
  <rect class="box-teal" x="160" y="372" width="360" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="394" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Change detection &amp; quantification</text>
  <text class="lbl-sub" x="340" y="412" font-size="12"
        text-anchor="middle" dominant-baseline="central">Green loss · built-up gain · per-city stats</text>
  <text class="lbl-step" x="650" y="404" font-size="11" text-anchor="end">05</text>

  <line class="arr-line" x1="340" y1="428" x2="340" y2="460"/>

  <!-- Step 6: Outputs -->
  <rect class="box-gray" x="160" y="460" width="360" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-gray" x="340" y="482" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Reproducible outputs</text>
  <text class="lbl-sub" x="340" y="500" font-size="12"
        text-anchor="middle" dominant-baseline="central">Jupyter notebook · methodology note · COG exports</text>
  <text class="lbl-step" x="650" y="492" font-size="11" text-anchor="end">06</text>
</svg>

<hr/>

<h3>Tech stack</h3>

<div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.75rem;">
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Python</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Sentinel-2</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">STAC</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">pystac-client</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">stackstac</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">NDVI · NDBI · BUI</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">xarray</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Rasterio</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Cloud Optimized GeoTIFF</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Jupyter</span>
</div>

<hr/>

<h3>Outputs</h3>

<ul>
  <li><strong>Jupyter notebook</strong> — fully parametrized, runs on any city with a STAC source</li>
  <li><strong>Methodology note</strong> — documents composite generation, index computation, and change quantification approach</li>
  <li><strong>COG exports</strong> — annual NDVI and built-up rasters per city, ready for QGIS or downstream analysis</li>
</ul>

<hr/>

<div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:1.5rem;">
  <a href="https://github.com/vincentondeng"
     target="_blank"
     style="display:inline-flex; align-items:center; gap:8px; padding:8px 18px;
            background:#1D9E75; color:white; border-radius:6px;
            text-decoration:none; font-size:14px; font-weight:500;">
    View on GitHub
  </a>
</div>
