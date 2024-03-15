from flask import Flask, render_template, request, jsonify
from main import *
import pandas as pd

app = Flask(__name__)
data = pd.read_pickle('/home/alvin/Documents/Amazon-main-project/pickels/28k_apperal_data_preprocessed')

@app.route('/')
def index():

    unique_brands = data['brand'].unique()
    popular_brands = data[data['brand'].isin(unique_brands)].sample(n=30)

    unique_brands1 = data['asin'].unique()
    recommendaton = data[data['asin'].isin(unique_brands1)].sample(n=30)

    unique_brands2 = data['color'].unique()
    recommendaton1 = data[data['color'].isin(unique_brands2)].sample(n=30)

    return render_template('index.html',recomend=None,recommendaton1=recommendaton1,recommendaton=recommendaton, popular_brands=popular_brands, unique_brands=unique_brands)


def idf2_w2v_brand(brand_name, w1, w2, num_results):
    brand_indices = data[data['brand'] == brand_name].index

    if len(brand_indices) == 0:
        print("No products found for the specified brand. But you might also like...")
        brand_indices = [0]

    doc_id = brand_indices[0]

    idf_w2v_dist = pairwise_distances(w2v_title_weight, w2v_title_weight[doc_id].reshape(1, -1))
    ex_feat_dist = pairwise_distances(extra_features, extra_features[doc_id])
    pairwise_dist = (w1 * idf_w2v_dist + w2 * ex_feat_dist) / float(w1 + w2)

    indices = np.argsort(pairwise_dist.flatten())[:num_results]
    pdists = np.sort(pairwise_dist.flatten())[:num_results]
    df_indices = list(data.index[indices])

    # Create a list to store the recommendations
    recommendations = []

    for i in range(num_results):
        asin = data['asin'].loc[df_indices[i]]
        brand = data['brand'].loc[df_indices[i]]
        distance = pdists[i]
        image = data['medium_image_url'].loc[df_indices[i]]
        title = data['title'].loc[df_indices[i]]

        # Append the recommendation as a dictionary
        recommendations.append({'Title': title,'ASIN': asin, 'Brand': brand, 'Euclidean distance from input': distance, 'Image': image})


    return recommendations

@app.route('/predict', methods=['POST'])
def recommend():
    selected_brand = request.form["selected_brand"]
    recommendations = idf2_w2v_brand(selected_brand, w1=10, w2=5, num_results=30)

    return jsonify(recommendations)
    #return render_template('index.html',recomend=recomend, recommendaton1=recommendaton1, recommendaton=recommendaton,
                           #popular_brands=popular_brands, unique_brands=unique_brands)


if __name__ == '__main__':
    app.run(debug=True)
