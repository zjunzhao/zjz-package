# Apripri: an implementation of a priori algorithm

class Apriori:
    
    def __init__(self, baskets):
        # preprocess baskets, transfrom itemnames to integers
        itemname_set = set()
        self.itemname_to_id = {}
        self.id_to_itemname = []
        self.nr_item = 0
        self.baskets = []
        for basket in baskets:
            new_basket = []
            for item in basket:
                if item not in itemname_set:
                    itemname_set.add(item)
                    self.itemname_to_id[item] = self.nr_item
                    self.id_to_itemname.append(item)
                    self.nr_item += 1
                new_basket.append(self.itemname_to_id[item])
            self.baskets.append(new_basket)

    def get_frequent_items(self, support_threshold):
        # calculate frequent items and delete infrequent items in baskets
        # return list of 1-tuple contained a frequent item
        count_table = [0]*self.nr_item
        for basket in self.baskets:
            for item in basket:
                count_table[item] += 1
        frequent_items = [(i,) for i in range(self.nr_item) if count_table[i]>=support_threshold]
        baskets = []
        for basket in self.baskets:
            baskets.append([item for item in basket if count_table[item]>=support_threshold])
        self.baskets = baskets
        return frequent_items
                
    def get_frequent_itemsets(self, itemset_size, support_threshold):
        # calculate frequent itemsets with given size
        # return a dist, keys are frequent itemsets, values are supports
        def next_idx(idx, n):
            # calculate next combination indice
            l = len(idx)
            for i in range(l):
                if idx[l-1-i]<n-1-i:
                    idx[l-1-i] += 1
                    for j in range(l-i, l):
                        idx[j] = idx[j-1]+1
                    return idx
            return -1
        frequent_itemsets = self.get_frequent_items(support_threshold)
        for k in range(2, itemset_size+1):
            pre_fi_set = set(frequent_itemsets)
            count_table = {}
            needcount_itemsets = []
            for old_itemset in frequent_itemsets:
                for new_item in range(old_itemset[-1]+1, self.nr_item):
                    new_itemset = old_itemset+(new_item,)
                    need_count = True
                    for i in range(k):
                        if new_itemset[:i]+new_itemset[i+1:] not in pre_fi_set:
                            need_count = False
                            break
                    if need_count:
                        count_table[new_itemset] = 0
                        needcount_itemsets.append(new_itemset)
            needcount_itemsets = set(needcount_itemsets)
            for basket in self.baskets:
                n = len(basket)
                if n<k:
                    continue
                idx = list(range(k))
                while idx is not -1:
                    itemset = tuple([basket[i] for i in idx])
                    if itemset in needcount_itemsets:
                        count_table[itemset] += 1
                    idx = next_idx(idx, n)
            frequent_itemsets = [key for (key, value) in count_table.items() if value>=support_threshold]
        ret = {}
        for itemset in frequent_itemsets:
            key = []
            for item in itemset:
                key.append(self.id_to_itemname[item])
            ret[tuple(key)] = count_table[itemset]
        return ret

if __name__=='__main__':
    pass