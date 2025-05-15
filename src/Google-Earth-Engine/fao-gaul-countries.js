// Load GAUL Level 0 dataset (countries)
var gaul0 = ee.FeatureCollection("FAO/GAUL/2015/level0");

// Get distinct country names
var countries = gaul0.aggregate_array('ADM0_NAME').distinct();

// Print the list of country names
print('Countries in FAO GAUL:', countries.sort());

var jk = gaul0.filter(ee.Filter.eq('ADM0_NAME', 'Jammu and Kashmir'));

// Add it to the map with a color
Map.centerObject(jk, 6);
Map.addLayer(jk, {color: 'red'}, 'Jammu and Kashmir');

