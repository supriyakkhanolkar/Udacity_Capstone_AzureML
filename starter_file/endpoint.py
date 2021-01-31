import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = "http://54b8f15b-6d1c-4500-a474-b6a848c4cdca.southcentralus.azurecontainer.io/score"

# If the service is authenticated, set the key or token
key = "ZNiebYvtizda6KtGcsMdklYAPyGN4hou"

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
   "id":842517,
    "radius_mean":20.57,
    "texture_mean":17.77,
    "perimeter_mean":132.9,
    "area_mean":1326,
    "smoothness_mean":0.08474,
    "compactness_mean":0.07864,
    "concavity_mean":0.0869,
    "concave points_mean":0.07017,
    "symmetry_mean":0.1812,
    "fractal_dimension_mean":0.05667,
    "radius_se":0.5435,
    "texture_se":0.7339,
    "perimeter_se":3.398,
    "area_se":74.08,
    "smoothness_se":0.005225,
    "compactness_se":0.01308,
    "concavity_se":0.0186,
    "concave points_se":0.0134,
    "symmetry_se":0.01389,
    "fractal_dimension_se":0.003532,
    "radius_worst":24.99,
    "texture_worst":23.41,
    "perimeter_worst":158.8,
    "area_worst":1956,
    "smoothness_worst":0.1238,
    "compactness_worst":0.1866,
    "concavity_worst":0.2416,
    "concave points_worst":0.186,
    "symmetry_worst":0.275,
    "fractal_dimension_worst":0.08902
          },
      ]
    }
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())


