// =====================================================================
// DSC 190 – FAST version (interactive, no Drive export needed)
// Simple random sample of points across Kalimantan; every derived variable
// is packed into one integer "code" per point and the script prints the
// histogram of codes. Python then decodes the histogram into all tables/figures.
// Simple random points are self-weighting: each point = AOI area / N.
// =====================================================================
var lulc = ee.Image('projects/mapbiomas-public/assets/indonesia/lulc/collection4_1/mapbiomas_indonesia_collection41_coverage_v1');
var fire = ee.Image('projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1');
var N_POINTS = 6000;   // per batch; run several batches with different SEED and merge
var SEEDS = [1];        // several batches in parallel, each its own random point set
var WHAT = 'c2015';    // 'fig1', 'c2015' or 'c2019'

var provinces = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
  .filter(ee.Filter.stringContains('ADM1_NAME', 'Kalimantan'));
var aoi = provinces.geometry();

function lc(y)    { return lulc.select('classification_' + y); }
function burnY(y) { return fire.select('burned_area_' + y).unmask(0).gt(0).rename('burned'); }
function anyBurn(y0, y1) {
  var imgs = [];
  for (var y = y0; y <= y1; y++) imgs.push(burnY(y));
  return ee.ImageCollection(imgs).max();
}
// land-cover groups: 1 forest, 2 non-forest natural, 3 oil palm, 4 pulpwood, 5 other agriculture, 6 other
function grp(img) {
  return img.remap([3, 5, 6, 76, 4, 10, 11, 12, 13, 35, 9, 14, 15, 18, 19, 20, 21, 36, 39, 40, 41],
                   [1, 1, 1, 1,  2, 2,  2,  2,  2,  3,  4, 5,  5,  5,  5,  5,  5,  5,  5,  5,  5], 6);
}

// ---- Figure 1 data: per year, pre-fire group of burned points (0 = not burned) ----
var f1bands = [];
for (var y = 2001; y <= 2024; y++) f1bands.push(grp(lc(y - 1)).multiply(burnY(y)).rename('y' + y));
if (WHAT === 'fig1') SEEDS.forEach(function (sd) {
  var pp = ee.FeatureCollection.randomPoints({region: aoi, points: N_POINTS, seed: sd, maxError: 100});
  var ff = ee.Image.cat(f1bands).sampleRegions({collection: pp, scale: 30, tileScale: 4});
  var hh = {};
  for (var y3 = 2001; y3 <= 2024; y3++) hh['y' + y3] = ff.aggregate_histogram('y' + y3);
  ee.Dictionary({n: ff.size(), hist: hh}).evaluate(function (r, err) { print('FIG1_JSON_SEED' + sd, err ? ('ERROR ' + err) : JSON.stringify(r)); });
});

// ---- Cohort codes ----
function cohortCode(t) {
  var pre = grp(lc(t - 1));
  var eligible = anyBurn(t - 5, t - 1).eq(0).and(pre.lte(2));
  var proj300 = lc(t - 1).projection().atScale(300);
  var dist = lc(t - 1).eq(35).reproject(proj300).fastDistanceTransform(64).sqrt()
    .reproject(proj300).multiply(300);
  var dband = ee.Image(3).where(dist.lt(15000), 2).where(dist.lt(5000), 1).where(dist.lt(2000), 0);
  var palm = lc(t + 4).eq(35).and(lc(t + 5).eq(35));
  var lagC = ee.Image(6), lagP = ee.Image(6), s = ee.Image(1);
  for (var k = 5; k >= 0; k--) {
    lagC = lagC.where(grp(lc(t + k)).gt(2), k);       // first year leaving natural vegetation
    s = s.and(lc(t + k).eq(35));
    lagP = lagP.where(s, k);                            // first year from which it stays oil palm
  }
  var code = pre.multiply(1e8)
    .add(grp(lc(t + 5)).multiply(1e7))
    .add(burnY(t).multiply(1e6))
    .add(anyBurn(t + 1, Math.min(t + 5, 2024)).multiply(1e5))
    .add(grp(lc(t - 5)).lte(2).multiply(1e4))
    .add(dband.multiply(1e3))
    .add(palm.multiply(100))
    .add(lagC.multiply(10))
    .add(lagP);
  return code.updateMask(eligible).toInt32().rename('code');
}
[2015, 2019].forEach(function (t) {
  if (WHAT !== 'c' + t) return;
  SEEDS.forEach(function (sd) {
    var p = ee.FeatureCollection.randomPoints({region: aoi, points: N_POINTS, seed: sd, maxError: 100});
    var smp = cohortCode(t).sampleRegions({collection: p, scale: 30, tileScale: 4});
    ee.Dictionary({cohort: t, n: smp.size(), hist: smp.aggregate_histogram('code')})
      .evaluate(function (r, err) { print('COHORT_JSON_' + t + '_SEED' + sd, err ? ('ERROR ' + err) : JSON.stringify(r)); });
  });
});
print('AOI_KM2', aoi.area(1000).divide(1e6));
