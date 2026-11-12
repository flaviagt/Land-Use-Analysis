// =====================================================================
// DSC 190 – Kalimantan post-fire land-cover trajectories
// Google Earth Engine Code Editor script (paste into code.earthengine.google.com)
//
// What it does
//   STEP 0  Prints checks (band names, region, land-cover classes present).
//   STEP 1  Export A: burned area (ha) per year 2001–2024 by PRE-fire class (t-1).
//   STEP 2  Export B: pixel counts per sampling stratum (needed for weights).
//   STEP 3  Export C: stratified random sample of pixels for cohorts 2015 & 2019,
//           with land cover t-1 … t+5, burn flags, distance to existing oil palm.
//
// Design (agreed defaults):
//   - "First-fire" cohort: burned in year t, NOT burned in t-5 … t-1.
//   - Comparison pool: not burned in t, also not burned in t-5 … t-1.
//   - Baseline = land cover in t-1 (avoids reading the fire scar as "before").
//   - Follow-up = t+1 … t+5. burned_later flag lets us test re-burn sensitivity.
// =====================================================================

// ---------- Inputs (asset IDs from MapBiomas user-toolkit) ----------
var lulc = ee.Image('projects/mapbiomas-public/assets/indonesia/lulc/collection4_1/mapbiomas_indonesia_collection41_coverage_v1');
var fire = ee.Image('projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1');

var COHORTS = [2015, 2019];     // add e.g. 2018 later as a non-El Niño comparison
var POINTS_PER_STRATUM = 2000;  // per cohort x burned x pre-fire class
var EXPORT_FOLDER = 'dsc190_kalimantan';

// ---------- Region: Kalimantan provinces from FAO GAUL ----------
var provinces = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
  .filter(ee.Filter.stringContains('ADM1_NAME', 'Kalimantan'));
var aoi = provinces.geometry();

// ---------- Helpers ----------
function lc(y)     { return lulc.select('classification_' + y); }
function burnY(y)  { return fire.select('burned_area_' + y).unmask(0).gt(0).rename('burned'); }  // same band name every year so collections can be combined
function anyBurn(y0, y1) {          // 1 if burned in any year y0..y1 (inclusive)
  var imgs = [];
  for (var y = y0; y <= y1; y++) imgs.push(burnY(y));
  return ee.ImageCollection(imgs).max();
}

// =====================================================================
// STEP 0 – CHECKS (look at the Console before running exports)
// =====================================================================
print('LULC bands (expect classification_1988 … _2024):', lulc.bandNames());
print('Fire bands (expect burned_area_2000 … _2024):', fire.bandNames());
print('Provinces found (expect 4–5 Kalimantan provinces):', provinces.aggregate_array('ADM1_NAME'));
print('AOI area km2 (Kalimantan is roughly 530–550k):', aoi.area(1000).divide(1e6));
print('Fire pixel values in 2015 (what does "burned" look like?):',
  fire.select('burned_area_2015').reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(), geometry: aoi, scale: 1000,
    maxPixels: 1e10, bestEffort: true}));
print('Land-cover classes present in 2014 (code: pixel count @1 km):',
  lc(2014).reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(), geometry: aoi, scale: 1000,
    maxPixels: 1e10, bestEffort: true}));

Map.centerObject(aoi, 6);
Map.addLayer(lc(2014).clip(aoi), {min: 0, max: 80, palette: ['white','green','yellow','red','blue']}, 'LULC 2014 (rough colors)', false);
Map.addLayer(burnY(2015).selfMask().clip(aoi), {palette: ['orange']}, 'Burned 2015');
Map.addLayer(provinces.style({color: 'black', fillColor: '00000000'}), {}, 'Provinces');

// =====================================================================
// STEP 1 – Export A: burned area per year by pre-fire land cover (t-1)
// =====================================================================
var areaRows = [];
for (var y = 2001; y <= 2024; y++) {
  var stats = ee.Image.pixelArea().divide(1e4).rename('ha')
    .updateMask(burnY(y))
    .addBands(lc(y - 1).rename('lc_prev'))
    .reduceRegion({
      reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'lc_prev'}),
      geometry: aoi, scale: 30, maxPixels: 1e13, tileScale: 8});
  var yy = y;  // capture
  areaRows.push(ee.FeatureCollection(ee.List(stats.get('groups')).map(function (g) {
    g = ee.Dictionary(g);
    return ee.Feature(null, {year: yy, lc_prev: g.get('lc_prev'), burned_ha: g.get('sum')});
  })));
}
Export.table.toDrive({
  collection: ee.FeatureCollection(areaRows).flatten(),
  description: 'A_burned_area_by_prefire_class',
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['year', 'lc_prev', 'burned_ha']});

// =====================================================================
// STEP 2 & 3 – cohort images, stratum counts, stratified sample
// =====================================================================
function cohortImage(t) {
  var burned = burnY(t);
  var eligible = anyBurn(t - 5, t - 1).eq(0);          // first-fire / clean history
  var lcPrev = lc(t - 1);
  // distance (m) to nearest oil palm (class 35) in t-1, capped at ~15 km
  var distPalm = lcPrev.eq(35).fastDistanceTransform(512).sqrt().multiply(30)
    .min(15360).unmask(15360).rename('dist_palm_m');
  var bands = [
    burned.multiply(1000).add(lcPrev).rename('stratum'),
    burned.rename('burned'),
    anyBurn(t + 1, Math.min(t + 5, 2024)).rename('burned_later'),
    distPalm,
    lc(t - 5).rename('lc_m5'),                          // 5 yrs before: stable vs recently cleared land
    lc(t - 1).rename('lc_m1'), lc(t).rename('lc_0')];
  for (var k = 1; k <= 5; k++) bands.push(lc(t + k).rename('lc_p' + k));
  return ee.Image.cat(bands).updateMask(eligible).toInt();
}

var countRows = [];
var samples = [];
COHORTS.forEach(function (t) {
  var img = cohortImage(t);

  // STEP 2: population size of every stratum (for sampling weights)
  var hist = ee.Dictionary(img.select('stratum').reduceRegion({
    reducer: ee.Reducer.frequencyHistogram(), geometry: aoi, scale: 30,
    maxPixels: 1e13, tileScale: 16}).get('stratum'));
  countRows.push(ee.FeatureCollection(hist.keys().map(function (k) {
    return ee.Feature(null, {cohort: t, stratum: ee.Number.parse(k), n_pixels: hist.get(k)});
  })));

  // STEP 3: equal-size random sample within each stratum
  var s = img.stratifiedSample({
    numPoints: POINTS_PER_STRATUM, classBand: 'stratum', region: aoi,
    scale: 30, seed: t, geometries: true, tileScale: 16, dropNulls: true});
  samples.push(s.map(function (f) {
    var c = f.geometry().coordinates();
    return f.set({cohort: t, lon: c.get(0), lat: c.get(1)});
  }));
});

Export.table.toDrive({
  collection: ee.FeatureCollection(countRows).flatten(),
  description: 'B_stratum_pixel_counts',
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['cohort', 'stratum', 'n_pixels']});

Export.table.toDrive({
  collection: ee.FeatureCollection(samples).flatten(),
  description: 'C_cohort_sample',
  folder: EXPORT_FOLDER, fileFormat: 'CSV',
  selectors: ['cohort', 'stratum', 'burned', 'burned_later', 'dist_palm_m',
              'lc_m5', 'lc_m1', 'lc_0', 'lc_p1', 'lc_p2', 'lc_p3', 'lc_p4', 'lc_p5',
              'lon', 'lat']});
