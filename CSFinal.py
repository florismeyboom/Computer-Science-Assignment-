# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:56:15 2023

@author: meybo
"""
import json
import random
import math
import pandas as pd
import numpy as np
import time
import re
from sklearn.utils import resample
from sklearn.metrics import f1_score, jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from random import shuffle
import matplotlib.pyplot as plt

# Load the JSON file into a DataFrame
#file = "TVs-all-merged.json"
with open('/Users/meybo/OneDrive/Documenten/Master Econometrie/Blok 2/Computer science/TVs-all-merged.json', 'r') as file:
    dataset = json.load(file)

# put all data in a list so that index order is fixed


def toList(dataset):
    temp = list(dataset.values())
    fin = []
    for i in temp:
        for j in i:
            fin.append(j)
    return fin

brands = ['Panasonic', 'Samsung', 'Sharp', 'Coby', 'LG', 'Sony',
        'Vizio', 'Dynex', 'Toshiba', 'HP', 'Supersonic', 'Elo',
        'Proscan', 'Westinghouse', 'SunBriteTV', 'Insignia', 'Haier',
        'Pyle', 'RCA', 'Hisense', 'Hannspree', 'ViewSonic', 'TCL',
        'Contec', 'NEC', 'Naxa', 'Elite', 'Venturer', 'Philips',
        'Open Box', 'Seiki', 'GPX', 'Magnavox', 'Hello Kitty', 'Naxa', 'Sanyo',
        'Sansui', 'Avue', 'JVC', 'Optoma', 'Sceptre', 'Mitsubishi', 'CurtisYoung', 'Compaq',
        'UpStar', 'Azend', 'Contex', 'Affinity', 'Hiteker', 'Epson', 'Viore', 'VIZIO','SIGMAC', 'Craig','ProScan', 'Apple']

brands = [brand.lower() for brand in brands]  # Convert brand names to lowercase

resols = ["720p", "1080p", "4K"]

# Calculate the total number of items in the dataset
total_items = sum(len(dataset[key]) for key in dataset.keys())

# Initialize tvBrand and tvResol with the correct size
tvBrand = np.zeros(total_items)
tvResol = np.zeros(total_items)


index = 0
for key in dataset.keys():
    for item in dataset[key]:
        title = item.get('title', '').lower()  # Convert title to lower case for comparison

        # Check for brand in title
        for j, brand in enumerate(brands):
            if brand in title:
                tvBrand[index] = j
                break

        # Check for resolution in title
        for j, resol in enumerate(resols):
            if resol in title:
                tvResol[index] = j
                break
        
        index += 1


def clean_text(text):
    # Normalize units (Hertz and Inch)
    text = re.sub(r"(Hertz|hertz|Hz|HZ| hz|-hz|hz|/hz)", "hz", text)
    text = re.sub(r'("|\bInch\b|\b-inch\b|\binch\b|\binches\b)', "inch", text)
    text = re.sub(r'(Diagonal|Diagonal Size|diagonal|diag.|diagonally)', 'Diag.', text)

    # Convert to lower case
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[\/#&;]", "", text)

    # Remove specific words
    to_remove = ['newegg', 'neweggcom', 'bestbuycom', 'thenerds.net', 'nerds', 'buy', 'best', 'bestbuy',
                 'amazon', 'com', 'amazoncom', 'tv', 'the', 'newegg.com', 'amazon.com', 'bestbuy.com']
    words = text.split()
    words = [word for word in words if word not in to_remove]

    return " ".join(words)


# Apply the cleaning function
for product_id, product_list in dataset.items():
    for product in product_list:
        # Clean the 'title'
        if 'title' in product:
            product['title'] = clean_text(product['title'])

        # Clean attributes in 'featuresMap'
        if 'featuresMap' in product:
            for key, value in product['featuresMap'].items():
                product['featuresMap'][key] = clean_text(value)


# clean the data and extract all the model words.
df = pd.DataFrame()
keys = []
model_id = []
shop = []
titles = []
featuresMap = []

for key in dataset.keys():
    for i in range(len(dataset[key])):
        keys.append(key)
        titles.append(clean_text(dataset[key][i]['title']))
        model_id.append(dataset[key][i]['modelID'])
        shop.append(dataset[key][i]['shop'])
        listfeatures = []
        values = [clean_text(item)[0]
                  for item in dataset[key][i]['featuresMap'].values()]
        for value in values:
            listfeatures.append(value)
        featuresMap.append(listfeatures)
       
        
def model_words(titles, featuresMap):
    MWtitles = []
    for i in range(len(titles)):
        MW = re.findall('((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', str(titles[i]))
        MWtitles.append(MW)
    
    MWpairs = []
    MWpairs2 = []
    for i in range(len(featuresMap)):
        # results = re.findall("(ˆ\d+(\.\d+)?[a-zA-Z]+$|ˆ\d+(\.\d+)?$)", str(featuresMap[i]))
        MW2 = re.findall("((?:[a-zA-Z]+[\x21-\x7E]+[0-9]|[0-9]+[\x21-\x7E]+[a-zA-Z])[a-zA-Z0-9]*)", str(featuresMap[i]))
        MWpairs2.append(MW2)
    
    for i in range(len(MWpairs2)):
        result = re.findall(r'\d+', str(MWpairs2[i]))
        MWpairs.append(result)
    
    titleplusfeatures = []
    for i in range(len(titles)):
        titleplusfeatures.append(MWtitles[i] + MWpairs[i])
    
    allmodelwords2 = []
    allmodelwords = []
    for i in range(len(titleplusfeatures)):
        # results1 = re.findall(" (\d+\.?\d+) "  , str(combinedtitlefeature[i]))
        results = (titleplusfeatures[i])
        allmodelwords2.append(results)
        # results2 = re.findall('[a-zA-Z\s]', str(titlewithmodelwords[i]))
        # titlewithmodelwords.remove(results2)
    # ("(\d+\.\d+) +$| r'\d+'$"
    for i in range(len(allmodelwords2)):
        result = list(dict.fromkeys(allmodelwords2[i]))
        allmodelwords.append(result)
        
    return allmodelwords

def process_model_words(titles, featuresMap):
    MWtitles = []
    MWpairs = []

    # Extract model words from titles
    for title in titles:
        MW = re.findall(
            '((?:[a-zA-Z]+[0-9]|[0-9]+[a-zA-Z])[a-zA-Z0-9]*)', title)
        MWtitles.append(MW)

    for feature_list in featuresMap:
        feature_words = []
        for feature in feature_list:
            if isinstance(feature, str):
                MW2 = re.findall(
                    "((?:[a-zA-Z]+[\x21-\x7E]+[0-9]|[0-9]+[\x21-\x7E]+[a-zA-Z])[a-zA-Z0-9]*)", feature)
                numeric_parts = re.findall(r'\d+', str(MW2))
                feature_words.extend(numeric_parts)
        MWpairs.append(feature_words)

    # Combine and deduplicate model words from titles and featuresMap
    allmodelwords = []
    for i in range(len(titles)):
        combined_model_words = list(set(MWtitles[i] + MWpairs[i]))
        allmodelwords.append(combined_model_words)

    return allmodelwords


# is walking through all the elements from inputData and it breaks them in subsets of k elements
def createShingle(inputData, k):
    shingles = []
    for inputRow in inputData:
        shingle = set()
        for i in range(len(inputRow) - k + 1):
            # Convert the shingle to a tuple before adding to the set
            shingle.add(tuple(inputRow[i:i + k]))
        shingles.append(shingle)
    return shingles

# in vocabulary I will have all the shingles
def createVocabulary(shingles):
    vocabulary = shingles[0]
    for i in range(1, len(shingles)):
        vocabulary = vocabulary.union(shingles[i])
    return set(vocabulary)


def createBinaryVectors(vocabulary, shingles):
    binaryVectors = []
    for shingle in shingles:
        # this kind of vector will have the same lenght as a shingle
        # is walking through all the elements from a shingle and if that element is also in the vocabulary, then it will generate 1 in the coresponding position in the vector; otherwise it will generate 0
        vector = [1 if x in shingle else 0 for x in vocabulary]
        binaryVectors.append(vector)
    return binaryVectors

# I call it in the function buildMinhashFunc - under
def createHashVector(size):
    hashList = list(range(size))  # Range from 0 to size-1
    shuffle(hashList)
    return hashList


# noVectors =  how many vectors hash will create - is a random number - the bigger the number the higher the accuracy
# vocabularySize = how many elements are in the vocabulary
def buildMinhashVectors(vocabularySize, noVectors):
    hashes = []
    for _ in range(noVectors):
        hashVector = createHashVector(vocabularySize)
        if len(hashVector) != vocabularySize:
            print("Hash vector length mismatch detected")
        hashes.append(hashVector)
    return hashes


# function to create signatures = the process of converting binaryVectors into dense vectors
def createSignatures(minhashVectors, vocabulary, binaryVectors, num_perm):
    signatures = []
    for binaryVector in binaryVectors:
        signature = []
        for minhashVector in minhashVectors:
            min_hash = len(vocabulary)  # Default value for min_hash
            for i in range(len(vocabulary)):
                index = minhashVector[i]
                if binaryVector[index] == 1:
                    min_hash = i
                    break
            signature.append(min_hash)
        
        # Ensure the signature length matches num_perm
        if len(signature) != num_perm:
            print("Signature length mismatch detected")
            signature.extend([len(vocabulary)] * (num_perm - len(signature)))
        
        signatures.append(signature)

    return signatures

def generate_hash_function(n_rows, max_shingle_id):
    # Generate random hash coefficients
    a = np.random.randint(0, max_shingle_id, n_rows)
    b = np.random.randint(0, max_shingle_id, n_rows)
    return a, b

def apply_lsh(signatures, num_bands, max_shingle_id, train_indices):
    n_rows = len(signatures[0])
    rows_per_band = n_rows // num_bands
    potential_pairs = set()

    for band in range(num_bands):
        a, b = generate_hash_function(rows_per_band, max_shingle_id)
        buckets = defaultdict(list)

        for idx in train_indices:  # Only consider indices from the training set
            signature = signatures[idx]
            slice_signature = signature[band * rows_per_band : (band + 1) * rows_per_band]
            hash_value = sum((a * slice_signature + b) % max_shingle_id)

            buckets[hash_value].append(idx)

        for bucket_items in buckets.values():
            if len(bucket_items) > 1:
                for i in range(len(bucket_items)):
                    for j in range(i + 1, len(bucket_items)):
                        potential_pairs.add((bucket_items[i], bucket_items[j]))

    print("potential pairs LSH found =", len(potential_pairs))
    return potential_pairs

def filter_pairs_by_brand_and_resol(pairs, brands, resols):
    filtered_pairs = []
    for pair in pairs:
        item1, item2 = pair
        #Check if both items have brand and resolution information
        have_brand_info = brands[item1] != 0 and brands[item2] != 0
        have_resol_info = resols[item1] != 0 and resols[item2] != 0

        # Keep the pair if both items have the same brand and resolution, or if one of them lacks this information
        if (not have_brand_info or brands[item1] == brands[item2]) and (not have_resol_info or resols[item1] == resols[item2]):
            filtered_pairs.append(pair)
    
    # Print the number of pairs left after filtering
    print(f"Number of pairs after filtering: {len(filtered_pairs)}")

    return filtered_pairs


def jaccardDistance(shWord1, shWord2):
    return len(shWord1.intersection(shWord2)) / (len(shWord1.union(shWord2))+0.0001)

def jaccard_distance(set1, set2):
    # Calculate the intersection and union of the two sets
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # Jaccard similarity coefficient is the size of intersection divided by size of union
    jaccard_similarity = len(intersection) / len(union) if union else 1
    
    # Jaccard distance is 1 minus the similarity coefficient
    return 1 - jaccard_similarity


def predict_duplicates(pairs, shingles, shops, similarity_threshold):
    predicted_duplicates = []
    similarity_samples = []  # Collect some similarity scores for debugging

    for pair in pairs:
        item1, item2 = pair
        # Skip pairs from the same shop
        if shops[item1] == shops[item2]:
            continue
        shingle1 = shingles[item1]
        shingle2 = shingles[item2]
        similarity = jaccardDistance(shingle1, shingle2)
        if similarity >= similarity_threshold:
            predicted_duplicates.append(pair)
        similarity_samples.append(similarity)

    return predicted_duplicates


def trueduplicates(data_cleaned):
    trueDuplicates = []
    key_indices = {}

    for i, key in enumerate(data_cleaned['modelID']):
        if key in key_indices:
            for j in key_indices[key]:
                trueDuplicates.append((j, i))
            key_indices[key].append(i)
        else:
            key_indices[key] = [i]

    return trueDuplicates

def createCandidateMatrix(predicted_duplicates, dataset_size):
    # Initialize a matrix of zeros with dimensions equal to the dataset size
    candidate_matrix = np.zeros((dataset_size, dataset_size), dtype=int)

    # Mark predicted duplicate pairs in the matrix
    for pair in predicted_duplicates:
        i, j = pair  # Assuming pair is a tuple like (index1, index2)
        candidate_matrix[i, j] = 1
        candidate_matrix[j, i] = 1  # Mark both (i, j) and (j, i)

    return candidate_matrix

def create_dissimilarity_matrix(predicted_pairs, processed_model_words, dataset_size):
    dissim_matrix = np.zeros((dataset_size, dataset_size))

    for pair in predicted_pairs:
        item1, item2 = pair
        set1 = set(processed_model_words[item1])
        set2 = set(processed_model_words[item2])

        # Handle empty sets to avoid division by zero
        if len(set1) == 0 and len(set2) == 0:
            dissimilarity = 1  # Maximum dissimilarity
        else:
            union_size = len(set1.union(set2))
            if union_size == 0:
                dissimilarity = 1  # Handle cases where the union might be zero
            else:
                dissimilarity = 1 - len(set1.intersection(set2)) / union_size

        dissim_matrix[item1, item2] = dissim_matrix[item2, item1] = dissimilarity

    return dissim_matrix



def linkage_clustering(dissimilarity_matrix, threshold):
    n_products = len(dissimilarity_matrix)

    # Initialize AgglomerativeClustering with precomputed dissimilarity matrix
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage='average', distance_threshold=threshold)

    # Fit the clustering model
    clusters = clustering.fit_predict(dissimilarity_matrix)

    # Mapping product indices to cluster labels
    clusters_dict = {}
    for idx, cluster_label in enumerate(clusters):
        if cluster_label not in clusters_dict:
            clusters_dict[cluster_label] = [idx]
        else:
            clusters_dict[cluster_label].append(idx)

    # Extract duplicate pairs from the clusters
    duplicate_pairs = []
    for cluster_indices in clusters_dict.values():
        if len(cluster_indices) > 1:
            duplicate_pairs.extend([(i, j) for i in cluster_indices for j in cluster_indices if i < j])

    return duplicate_pairs


def performance(pred_dupes, real_dupes, classdupes, candidate, data):

    nPredDupes = len(pred_dupes)
    nRealDupes = len(real_dupes)
    nClassDupes = len(classdupes)

    TP = []
    FP = []
    for i in range(0, nPredDupes):
        if pred_dupes[i] in real_dupes:
            TP.append(pred_dupes[i])
        else:
            FP.append(pred_dupes[i])
        
    TPC = []
    FPC = []    
    for i in range(0, nClassDupes):
        if classdupes[i] in real_dupes:
            TPC.append(classdupes[i])
        else:
            FPC.append(classdupes[i])

    nTPC = len(TPC)
    nFPC = len(FPC)/1
    nFNC = (nRealDupes - len(TPC))



    nTP = len(TP)
    nFP = len(FP)/1
    nFN = (nRealDupes - len(TP))

    nComps = np.count_nonzero(candidate)/2
    print("nComps = ", nComps)
    nCompsPossible = len(data)*(len(data)-1) * 0.5

    comparisonFrac = nComps/nCompsPossible

    PQ = nTP/(nComps+0.0001)
    PC = nTP/(nRealDupes+0.0001)

    # precision = nTP/(nTP+nFP+0.0001)
    # recall = nTP/(nTP + nFN+0.0001)
    
    precision = nTPC/(nTPC+nFPC+0.0001)
    recall = nTPC/(nTPC + nFNC+0.0001)

    def F1(precision, recall):
        f1 = (2*precision*recall)/(precision+recall+0.0001)
        return f1

    def F1Star(PQ, PC):
        f1star = (2*PQ*PC)/(PQ+PC+0.0001)
        return f1star

    F1 = F1(precision, recall)
    F1Star = F1Star(PQ, PC)

    return nTP, PQ, PC, F1, F1Star, comparisonFrac

def perf(pred_dupes, real_dupes, candidate, data):

    nPredDupes = len(pred_dupes)
    nRealDupes = min(len(real_dupes),0.63*399)

    TP = []
    FP = []
    for i in range(0, nPredDupes):
        if pred_dupes[i] in real_dupes:
            TP.append(pred_dupes[i])
        else:
            FP.append(pred_dupes[i])
        
    nTP = len(TP)
    nFP = len(FP)/1
    nFN = (nRealDupes - len(TP))

    nComps = np.count_nonzero(candidate)/2
    print("nComps = ", nComps)
    nCompsPossible = len(data)*(len(data)-1) * 0.5

    comparisonFrac = nComps/nCompsPossible

    # precision = nTP/(nTP+nFP+0.0001)
    # recall = nTP/(nTP + nFN+0.0001)
    
    precision = nTP/(nTP+nFP+0.0001)
    recall = nTP/(nTP + nFN+0.0001)

    def F1(precision, recall):
        f1 = (2*precision*recall)/(precision+recall+0.0001)
        return f1

    F1 = F1(precision, recall)

    return nTP, F1, comparisonFrac

def perfstar(pred_dupes, real_dupes, candidate, data):

    nPredDupes = len(pred_dupes)
    nRealDupes = min(len(real_dupes),0.63*399)

    TP = []
    FP = []
    for i in range(0, nPredDupes):
        if pred_dupes[i] in real_dupes:
            TP.append(pred_dupes[i])
        else:
            FP.append(pred_dupes[i])
        
    nTP = len(TP)
    nFP = len(FP)/1
    nFN = (nRealDupes - len(TP))

    nComps = np.count_nonzero(candidate)/2
    print("nComps star = ", nComps)
    nCompsPossible = len(data)*(len(data)-1) * 0.5

    comparisonFrac = nComps/nCompsPossible

    # precision = nTP/(nTP+nFP+0.0001)
    # recall = nTP/(nTP + nFN+0.0001)
    
    PQ = nTP/(nComps+0.0001)
    PC = nTP/(nRealDupes+0.0001)

    def F1Star(PQ, PC):
         f1star = (2*PQ*PC)/(PQ+PC+0.0001)
         return f1star

    F1Star = F1Star(PQ, PC)

    return nTP, PQ, PC,  F1Star, comparisonFrac

#%%

num_bootstraps = 1
band_sizes = [5, 10, 25, 50, 60, 75, 100]

metrics = defaultdict(lambda: {'fractions_of_comparisons': [], 'fractions_of_comparisonsStar': [], 'pair_completeness_values': [], 'pair_quality_values': [], 'f1_star_scores': [], 'f1_scores': []})


for bootstrap_iteration in range(num_bootstraps):
    print(f"Bootstrap Iteration: {bootstrap_iteration + 1}")
    for bands in band_sizes:
        bootstrap_sample = resample(toList(dataset), replace=True, n_samples=len(dataset))
        
        # Split into training and test sets (63% train, 37% test)
        split_index = int(0.63 * len(bootstrap_sample))
        train_set = bootstrap_sample[:split_index]
        test_set = bootstrap_sample[split_index:]
        train_df = pd.DataFrame(train_set)
        test_df = pd.DataFrame(test_set)
        
        train_items_count = len(train_set)
        tvBrand1 = np.zeros(train_items_count)
        tvResol1 = np.zeros(train_items_count)
        
        # Populate tvBrand and tvResol using train_set
        for index, item in enumerate(train_set):
            title = item.get('title', '').lower()  # Convert title to lower case for comparison
        
            # Check for brand in title
            for j, brand in enumerate(brands):
                if brand in title:
                    tvBrand1[index] = j
                    break
        
            # Check for resolution in title
            for j, resol in enumerate(resols):
                if resol in title:
                    tvResol1[index] = j
                    break
        
        train_model_ids = [item['modelID'] for item in train_set]
        train_titles = [item['title'] for item in train_set]
        train_featuresMap = [item['featuresMap'] for item in train_set]
        train_shop = [item['shop'] for item in train_set]
        
        allmodelwords = process_model_words(train_titles, train_featuresMap)
        
        print("number of bands = ", bands)
        k = 5  # Shingle length
        noVectors = 1500  # Number of MinHash vectors
        num_perm = noVectors
        #bands = 20  # Number of bands
        r = noVectors//bands   # Rows per band
        
        shingles = createShingle(allmodelwords, k)
        vocabulary = createVocabulary(shingles)
        binaryVectors = createBinaryVectors(vocabulary, shingles)
        
        minhashVectors = buildMinhashVectors(len(vocabulary), noVectors)
        signatures = createSignatures(minhashVectors, vocabulary, binaryVectors, num_perm)
        train_indices = list(range(len(train_titles)))
        signatures_array = np.array([signatures[i] for i in train_indices]).T
           
        a,b = generate_hash_function(num_perm, 5003)    
        lsh_pairs = apply_lsh(signatures_array, bands, 5003, train_indices)
        filtered_pairs = filter_pairs_by_brand_and_resol(lsh_pairs, tvBrand1, tvResol1)
        
            
        class_duplicates = predict_duplicates(filtered_pairs, shingles, shop, 0.0001)
        predicted_duplicates = predict_duplicates(filtered_pairs, shingles, train_shop, 0)
        candidate_matrix = createCandidateMatrix(predicted_duplicates, len(allmodelwords))
        candidate_matrixClass = createCandidateMatrix(class_duplicates, len(allmodelwords))
      
        data_frame = pd.DataFrame({'modelID': train_model_ids})
        #data_frame = pd.DataFrame({'modelID': model_id})
        true_duplicates = trueduplicates(data_frame)
        #nTP, PQ, PC, F1, F1Star, comparisonFrac = performance(predicted_duplicates, true_duplicates, class_duplicates, candidate_matrix, data_frame)
        
        nTP, F1, comparisonFrac = perf(class_duplicates, true_duplicates, candidate_matrixClass, data_frame)
        nTPstar, PQ, PC, F1Star, comparisonFracStar = perfstar(predicted_duplicates, true_duplicates, candidate_matrix, data_frame)
        

        
        print("total true = ", len(true_duplicates))
        print(f"True Positives: {nTP}")
        print(f"True Positives star: {nTPstar}")
        print(f"Pair Quality: {PQ}")
        print(f"Pair Completeness: {PC}")
        print(f"F1 Score: {F1}")
        print(f"F1* Score: {F1Star}")
        print(f"Comparison Fraction: {comparisonFrac}")
        print(f"Comparison Fraction star: {comparisonFracStar}")
        print("")
        
average_metrics = {band: {metric: np.mean(values) for metric, values in band_metrics.items()} for band, band_metrics in metrics.items()}

fractions_of_comparisonsStar_avg = [average_metrics[band]['fractions_of_comparisonsStar'] for band in band_sizes]
fractions_of_comparisons_avg = [average_metrics[band]['fractions_of_comparisons'] for band in band_sizes]
pair_completeness_avg = [average_metrics[band]['pair_completeness_values'] for band in band_sizes]
pair_quality_avg = [average_metrics[band]['pair_quality_values'] for band in band_sizes]
f1_star_avg = [average_metrics[band]['f1_star_scores'] for band in band_sizes]
f1_avg = [average_metrics[band]['f1_scores'] for band in band_sizes]

    
plt.plot(fractions_of_comparisonsStar_avg, pair_completeness_avg)
plt.xlabel("Fraction of Comparisons")
plt.ylabel("Pair Completeness")
plt.title("Pair Completeness vs Fraction of Comparisons")
plt.show()

# Plot Pair Quality vs Fraction of Comparisons
plt.plot(fractions_of_comparisonsStar_avg, pair_quality_avg)
plt.xlabel("Fraction of Comparisons")
plt.ylabel("Pair Quality")
plt.title("Pair Quality vs Fraction of Comparisons")
plt.show()

# Plot F1* Score vs Fraction of Comparisons
plt.plot(fractions_of_comparisonsStar_avg, f1_star_avg)
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1* Score")
plt.title("F1* Score vs Fraction of Comparisons")
plt.show()

# Plot F1 Score vs Fraction of Comparisons
plt.plot(fractions_of_comparisons_avg, f1_avg)
plt.xlabel("Fraction of Comparisons")
plt.ylabel("F1 Score")
plt.title("F1 Score vs Fraction of Comparisons")
plt.show()



