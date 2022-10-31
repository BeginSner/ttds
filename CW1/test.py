from preprocessing import *
from searching_func import *




if __name__ == "__main__":
    trec_path = 'collections\\trec.5000.xml'
    index_path = 'collections\\index'
    dic = load_xml(trec_path)
    terms = find_uniq(dic)
    collection_dic = create_position_index(terms)
    save_position_index(collection_dic, path=index_path)

    with open(trec_path, 'r', encoding='utf-8') as f:
        xml_file = f.read()
    # load total text in the dict_file
    dict_file = xmltodict.parse(xml_file)
    dict_file = dict_file['document']['DOC']
    N = len(dict_file) # calculate N

    # testing search
    with open(index_path+'.dat', 'rb') as f:
        data = pickle.load(f)

    query_path = 'queries.boolean.txt'
    query_results = query_search(query_path, data, N)

    result_path = 'tfidf.results'
    write_query_results(query_results, result_path)
