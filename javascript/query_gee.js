

/*
//ADD LANDCOVER 

var dataset = ee.Image("COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019")
.select('discrete_classification');

Map.setCenter(-88.6, 26.4, 1);

Map.addLayer(dataset, {}, "Land Cover");

var landcov_discrete = dataset.select('discrete_classification')

var lcdIC = ee.ImageCollection(landcov_discrete)
var mapfunc = function(feat) {
  var geom = feat.geometry()
  var addProp = function(img, f) {
    var newf = ee.Feature(f)
    var date = img.date().format()
    var value = img.reduceRegion(ee.Reducer.first(), geom, 30).get("discrete_classification")
    return ee.Feature(ee.Algorithms.If(value, newf.set(date, value), newf.set(date, "NaN")))

  }
  
  var newFeat = ee.Feature(lcdIC.iterate(addProp, feat))
  return newFeat

}

var newft = table4.map(mapfunc);
Export.table.toDrive(newft, 
"export_Points", 
"export_Points", 
"export_Points");
*/

//ADD Elevations

/*


var dataset = ee.Image("USGS/GMTED2010")
 .select('be75');

Map.setCenter(-88.6, 26.4, 1);

Map.addLayer(dataset, {}, "Elevation");

var elev = dataset.select('be75');

var elevIC = ee.ImageCollection(elev);
var mapfunc = function(feat) {
    var geom = feat.geometry()
    var addProp = function(img, f) {
      var newf = ee.Feature(f)
      var date = "elev"
      var value = img.reduceRegion(ee.Reducer.first(), geom, 30).get("be75")
      return ee.Feature(ee.Algorithms.If(value, newf.set(date, value), newf.set(date, "NaN")))
  }
    var newFeat = ee.Feature(elevIC.iterate(addProp, feat))
    return newFeat
}

var newft = table4.map(mapfunc);
Export.table.toDrive(newft, 
"export_Points", 
"export_Points", 
"export_Points");
*/


// ADD Landcover

/*
var dataset = ee.Image("CSP/ERGo/1_0/Global/SRTM_landforms")
 .select('constant');

 Map.setCenter(-88.6, 26.4, 1);

 Map.addLayer(dataset, {}, "Landform");

var lf = dataset.select('constant');

var lfIC = ee.ImageCollection(lf);
var mapfunc = function(feat) {
    var geom = feat.geometry()
    var addProp = function(img, f) {
      var newf = ee.Feature(f)
      var date = "elev"
      var value = img.reduceRegion(ee.Reducer.first(), geom, 30).get("constant")
      return ee.Feature(ee.Algorithms.If(value, newf.set(date, value), newf.set(date, "NaN")))
  }
    var newFeat = ee.Feature(lfIC.iterate(addProp, feat))
    return newFeat
}

var newft = table4.map(mapfunc);
Export.table.toDrive(newft, 
"export_Points", 
"export_Points", 
"export_Points");

*/