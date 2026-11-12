// DSC 190 – data behind exploration Figures 2 and 3 (run in the Earth Engine Code Editor).
// Figure 2: share of each 0.05° (~5.5 km) cell burned in 2015 and 2019 (30 m MapBiomas Fire pixels averaged up),
//           computed in 2°x2° tiles with sampleRectangle to stay under memory limits.
// Figure 3: land-cover group at 20,000 random points (5 seeds x 4,000) for every year 2000-2024.
var FIRE = ee.Image('projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1');
var LULC = ee.Image('projects/mapbiomas-public/assets/indonesia/lulc/collection4_1/mapbiomas_indonesia_collection41_coverage_v1');
var aoi = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
  .filter(ee.Filter.stringContains('ADM1_NAME', 'Kalimantan')).geometry();
var RES = 0.05;
var grid = ee.Projection('EPSG:4326').scale(RES, -RES).translate(108.5, 4.4);
var p30 = LULC.select('classification_2014').projection(), p300 = p30.atScale(300);
function twoStep(img) {   // 30 m -> 300 m -> 0.05 deg, mean at each step
  return img.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024}).reproject(p300)
            .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024}).reproject(grid);
}
function burnedFrac(y) { return twoStep(FIRE.select('burned_area_' + y).unmask(0).gt(0).toFloat().setDefaultProjection(p30)); }
var stack = ee.Image.cat([burnedFrac(2015).rename('b2015'), burnedFrac(2019).rename('b2019'),
                          ee.Image(1).clip(aoi).unmask(0).reproject(grid).rename('kal')]).unmask(-1);
for (var x = 108.5; x < 119.1; x += 2) for (var y = 4.4; y > -4.3; y -= 2) {
  var tile = ee.Geometry.Rectangle([x, Math.max(y - 2, -4.3), Math.min(x + 2, 119.1), y], null, false);
  print('TILE', x, y, stack.sampleRectangle({region: tile, defaultValue: -1}).toDictionary());
}
function grp(img) {
  return img.remap([3,5,6,76,4,10,11,12,13,35,9,14,15,18,19,20,21,36,39,40,41],
                   [1,1,1,1,2,2,2,2,2,3,4,5,5,5,5,5,5,5,5,5,5], 6);
}
var bands = [];
for (var yr = 2000; yr <= 2024; yr++) bands.push(grp(LULC.select('classification_' + yr)).rename('y' + yr));
[101, 102, 103, 104, 105].forEach(function (sd) {
  var pts = ee.FeatureCollection.randomPoints({region: aoi, points: 4000, seed: sd, maxError: 100});
  var s = ee.Image.cat(bands).sampleRegions({collection: pts, scale: 30, tileScale: 4});
  var h = {}; for (var yy = 2000; yy <= 2024; yy++) h['y' + yy] = s.aggregate_histogram('y' + yy);
  print('LANDCOVER_SEED' + sd, ee.Dictionary({n: s.size(), hist: h}));
});
