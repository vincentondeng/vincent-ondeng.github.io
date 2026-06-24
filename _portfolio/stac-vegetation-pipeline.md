---
title: "STAC Vegetation Index Pipeline"
excerpt: "A reusable Python module for querying any STAC catalog, lazily loading satellite imagery, computing vegetation indices, and exporting Cloud Optimized GeoTIFFs — parametrized by bounding box, date range, and cloud cover."
collection: portfolio
header:
  teaser: "/images/stac-pipeline-thumb.svg"
---

<p>
Satellite imagery is everywhere — but getting from a STAC catalog to a clean,
analysis-ready raster is still more friction than it should be. This module
removes that friction.
</p>

<p>
Built around <code>pystac-client</code> and <code>stackstac</code>, the pipeline
queries any STAC-compliant catalog, lazily loads only the imagery that intersects
your area of interest, and computes vegetation indices including NDVI, EVI, and
SAVI — without pulling unnecessary data into memory. Results are exported as
Cloud Optimized GeoTIFFs, making outputs immediately compatible with downstream
tools like QGIS, GDAL, and cloud-based raster APIs.
</p>

<p>
The entire workflow is parametrized — pass in a bounding box, date range, and
cloud cover threshold and the module handles the rest. Designed for
reproducibility and scale.
</p>

<hr/>

<h3>Pipeline</h3>

<svg width="100%" viewBox="0 0 680 520" role="img" xmlns="http://www.w3.org/2000/svg">
  <title>STAC Vegetation Index Pipeline</title>
  <desc>
    A flowchart showing the pipeline from user parameters through STAC catalog
    query, lazy image loading, band selection, vegetation index computation,
    and Cloud Optimized GeoTIFF export.
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
      .box-purple { fill: #EEEDFE; stroke: #534AB7; }
      .lbl-gray   { fill: #444441; font-family: sans-serif; }
      .lbl-teal   { fill: #085041; font-family: sans-serif; }
      .lbl-purple { fill: #3C3489; font-family: sans-serif; }
      .lbl-sub    { fill: #5F5E5A; font-family: sans-serif; }
      .arr-line   { stroke: #888780; stroke-width: 1.5; fill: none;
                    marker-end: url(#arrow); }
    </style>
  </defs>

  <!-- Step 1: User parameters -->
  <rect class="box-gray" x="220" y="20" width="240" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-gray" x="340" y="45" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">User parameters</text>
  <text class="lbl-sub" x="340" y="63" font-size="12"
        text-anchor="middle" dominant-baseline="central">Bounding box · date range · cloud cover</text>

  <!-- Arrow 1 -->
  <line class="arr-line" x1="340" y1="76" x2="340" y2="108"/>

  <!-- Step 2: STAC catalog query -->
  <rect class="box-teal" x="180" y="108" width="320" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="130" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">STAC catalog query</text>
  <text class="lbl-sub" x="340" y="148" font-size="12"
        text-anchor="middle" dominant-baseline="central">pystac-client · any STAC-compliant source</text>

  <!-- Arrow 2 -->
  <line class="arr-line" x1="340" y1="164" x2="340" y2="196"/>

  <!-- Step 3: Lazy image loading -->
  <rect class="box-teal" x="180" y="196" width="320" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="218" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Lazy image loading</text>
  <text class="lbl-sub" x="340" y="236" font-size="12"
        text-anchor="middle" dominant-baseline="central">stackstac · only AOI tiles loaded into memory</text>

  <!-- Arrows to bands -->
  <line class="arr-line" x1="300" y1="252" x2="140" y2="284"/>
  <line class="arr-line" x1="340" y1="252" x2="340" y2="284"/>
  <line class="arr-line" x1="380" y1="252" x2="540" y2="284"/>

  <!-- Step 4a: Red band -->
  <rect class="box-purple" x="60" y="284" width="160" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-purple" x="140" y="306" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Red band</text>
  <text class="lbl-sub" x="140" y="324" font-size="12"
        text-anchor="middle" dominant-baseline="central">B04 · 665 nm</text>

  <!-- Step 4b: NIR band -->
  <rect class="box-purple" x="260" y="284" width="160" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-purple" x="340" y="306" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">NIR band</text>
  <text class="lbl-sub" x="340" y="324" font-size="12"
        text-anchor="middle" dominant-baseline="central">B08 · 842 nm</text>

  <!-- Step 4c: SWIR band -->
  <rect class="box-purple" x="460" y="284" width="160" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-purple" x="540" y="306" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">SWIR band</text>
  <text class="lbl-sub" x="540" y="324" font-size="12"
        text-anchor="middle" dominant-baseline="central">B11 · for EVI / SAVI</text>

  <!-- Arrows to index computation -->
  <line class="arr-line" x1="140" y1="340" x2="280" y2="372"/>
  <line class="arr-line" x1="340" y1="340" x2="340" y2="372"/>
  <line class="arr-line" x1="540" y1="340" x2="400" y2="372"/>

  <!-- Step 5: Index computation -->
  <rect class="box-teal" x="180" y="372" width="320" height="56" rx="8" stroke-width="0.5"/>
  <text class="lbl-teal" x="340" y="394" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Vegetation index computation</text>
  <text class="lbl-sub" x="340" y="412" font-size="12"
        text-anchor="middle" dominant-baseline="central">NDVI · EVI · SAVI</text>

  <!-- Arrow 5 -->
  <line class="arr-line" x1="340" y1="428" x2="340" y2="460"/>

  <!-- Step 6: COG export -->
  <rect class="box-gray" x="180" y="460" width="320" height="44" rx="8" stroke-width="0.5"/>
  <text class="lbl-gray" x="340" y="482" font-size="14" font-weight="500"
        text-anchor="middle" dominant-baseline="central">Cloud Optimized GeoTIFF export</text>
</svg>

<hr/>

<h3>Tech stack</h3>

<div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:0.75rem;">
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Python</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">STAC</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">pystac-client</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">stackstac</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Rasterio</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">NDVI · EVI · SAVI</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">Cloud Optimized GeoTIFF</span>
  <span style="padding:4px 12px; border:1px solid #ccc; border-radius:999px; font-size:13px;">xarray</span>
</div>

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
