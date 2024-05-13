from operator import itemgetter
import psycopg2
import math
import pandas as pd
import numpy as np
import heapq
import matplotlib.pyplot as plt

class SeeDB:
    def __init__(self, aggregate_func_list, attribute_list, group_by_list):
        self.aggregate_func_list = aggregate_func_list
        self.attribute_list = attribute_list
        self.group_by_list = group_by_list
        self.views_utilities_ranges = {}
        self.top_5_largest_views = {}

    def prepare_mf_pairs(self):
        self.mf_pairs = []
        for i in self.aggregate_func_list:
            for j in self.attribute_list:
                self.mf_pairs.append((i, j))

    def connection(self):
        conn = psycopg2.connect(database = "censusincome",
                        user = "postgres",
                        host= 'localhost',
                        port = 5432)
        self.cur = conn.cursor()

    def calc_utility_ranges(self, views_names, views_target_dict, views_ref_dict, not_first_phase):
        for name in views_names:
            (a,mf) = name
            target_pd = views_target_dict[name]
            ref_pd = views_ref_dict[name]

            vals_target = np.array(target_pd[[mf]])
            sum_target = np.sum(vals_target, axis=0)

            p_dict = target_pd.set_index('a')[target_pd.columns[1]].to_dict()
            p_dict_normal = {k: 0.0 if float(sum_target[0]) == 0.0 else float(v)/float(sum_target[0]) for k, v in p_dict.items()}

            vals_target = np.array(ref_pd[[mf]])
            sum_ref = np.sum(vals_target, axis=0)

            q_dict = ref_pd.set_index('a')[target_pd.columns[1]].to_dict()
            q_dict_normal = {k: 0.0 if float(sum_ref[0]) == 0.0 else float(v)/float(sum_ref[0]) for k, v in q_dict.items()}

            utility_np = 0.0
            for (k,v) in p_dict_normal.items():
                p_x = v
                if k in q_dict_normal:
                    q_x = q_dict_normal[k]
                    curr_val = 0.0 if (q_x == 0.0 or p_x/q_x == 0.0) else p_x + math.log(p_x/q_x)
                    utility_np = utility_np + (0.0 if math.isnan(curr_val) else curr_val)

            if not_first_phase:
                (l,r) = self.views_utilities_ranges[name]
                self.views_utilities_ranges[name] = (min(l,utility_np), max(r,utility_np))
            else:
                self.views_utilities_ranges[name] = (99999999999999999999999, utility_np)

    def create_visualisations(self):
        for (category, statistic) in self.top_5_largest_views.keys():
            self.cur.execute("select {a}, {s} from ci_target group by {a};".format(a=category, s=statistic))
            data1 = self.cur.fetchall()
            self.cur.execute("select {a}, {s} from ci_reference group by {a};".format(a=category, s=statistic))
            data2 = self.cur.fetchall()

            target_data = dict(data1)
            reference_data = dict(data2)

            if not target_data or not reference_data:
                print("Data not available for plotting for ({}, {})".format(category, statistic))
                continue

            num_groups = max(len(target_data), len(reference_data))
            func_name = statistic.split('(')[0]
            measure = statistic.split('(')[1][:-1]
            group_by = category

            all_keys = set(target_data.keys()).union(reference_data.keys())
            means_target = [target_data.get(key, 0) for key in all_keys]
            means_reference = [reference_data.get(key, 0) for key in all_keys]

            fig, ax = plt.subplots()
            index = np.arange(num_groups)
            bar_width = 0.35
            opacity = 0.8

            rects1 = plt.bar(index, means_target, bar_width,alpha=opacity,color='b',label='married')

            rects2 = plt.bar(index + bar_width, means_reference, bar_width, alpha=opacity,color='g',label='unmarried')

            plt.xlabel('{}'.format(group_by))
            plt.ylabel('{}({})'.format(func_name, measure))
            plt.xticks(index + bar_width, all_keys, rotation=45)
            plt.legend()

            plt.tight_layout()
            plt.savefig("plot_" + category + "_" + statistic + ".jpg")


    def create_temp_tables(self, i, phase_fraction_target, phase_fraction_ref):
        self.cur.execute("create temp table temp_table_target_{i} as (select * from ci_target offset {curr_offset} limit {phase_fraction_target})".format(i=i,curr_offset=(i-1)*phase_fraction_target, phase_fraction_target=phase_fraction_target))
        self.cur.execute("create temp table temp_table_reference_{i} as (select * from ci_target offset {curr_offset} limit {phase_fraction_ref})".format(i=i,curr_offset=(i-1)*phase_fraction_ref, phase_fraction_ref=phase_fraction_ref))

    def split_dataset(self):
        self.connection()
        self.cur.execute("select count(*) from ci_target;")
        count_target = self.cur.fetchall()[0][0]
        self.cur.execute("select count(*) from ci_reference;")
        count_ref = self.cur.fetchall()[0][0]
        phase_fraction_target = math.ceil(count_target/10)
        phase_fraction_ref = math.ceil(count_ref/10)
        return (phase_fraction_target, phase_fraction_ref)
    def main_method(self):
        self.prepare_mf_pairs()
        views_target_dict = {}
        views_ref_dict = {}
        views_names = set()
        (phase_fraction_target, phase_fraction_ref) = self.split_dataset()

        mf_pairs_modf = ["{f}({m})".format(f=f,m=m) for (f,m) in self.mf_pairs]
        mf_pairs_concat = ", ".join(mf_pairs_modf) + ", count(*)"

        pd_cols = mf_pairs_modf.copy()
        pd_cols.insert(0, "a")
        pd_cols.append("count(*)")

        self.create_temp_tables(1, phase_fraction_target=phase_fraction_target, phase_fraction_ref=phase_fraction_ref)

        for a in self.group_by_list:
            self.cur.execute("select {a}, {mf_pairs_concat} from temp_table_target_1 group by {a} ".format(a=a, mf_pairs_concat=mf_pairs_concat))
            combined_view_target = self.cur.fetchall()
            self.cur.execute("select {a}, {mf_pairs_concat} from temp_table_reference_1 group by {a} ".format(a=a, mf_pairs_concat=mf_pairs_concat))
            combined_view_ref = self.cur.fetchall()

            combined_views_target_pd = pd.DataFrame(combined_view_target, columns=pd_cols)
            combined_views_ref_pd = pd.DataFrame(combined_view_ref, columns=pd_cols)

            for mf in mf_pairs_modf:
                k = (a, mf)
                views_names.add(k)
                views_target_dict[k] = combined_views_target_pd[['a',mf]]
                views_ref_dict[k] = combined_views_ref_pd[['a',mf]]

            views_names.add((a,"count(*)"))
            views_target_dict[(a,"count(*)")] = combined_views_target_pd[['a',"count(*)"]]
            views_ref_dict[(a,"count(*)")] = combined_views_ref_pd[['a',"count(*)"]]

        self.calc_utility_ranges(views_names=views_names, views_target_dict=views_target_dict, views_ref_dict=views_ref_dict, not_first_phase=False)

        self.cur.execute('drop table temp_table_target_1')
        self.cur.execute('drop table temp_table_reference_1')

        print("First iteration done......")

        for i in range(2,11):
            views_target_dict = {}
            views_ref_dict = {}
            self.create_temp_tables(i, phase_fraction_target, phase_fraction_ref)
            for a in self.group_by_list:
                mf_pairs_extract = [view_name[1] for view_name in views_names if view_name[0] == a]
                if mf_pairs_extract:
                    mf_pairs_extract_concat = ", ".join(mf_pairs_extract)
                    pd_cols_local = mf_pairs_extract.copy()
                    pd_cols_local.insert(0, "a")
                    self.cur.execute("select {a}, {mf_pairs_extract_concat} from temp_table_target_{i} group by {a} ".format(i=i, a=a, mf_pairs_extract_concat=mf_pairs_extract_concat))
                    combined_view_target_phased = self.cur.fetchall()
                    self.cur.execute("select {a}, {mf_pairs_extract_concat} from temp_table_reference_{i} group by {a} ".format(i=i, a=a, mf_pairs_extract_concat=mf_pairs_extract_concat))
                    combined_view_ref_phased = self.cur.fetchall()
                    combined_views_target_phased_pd = pd.DataFrame(combined_view_target_phased, columns=pd_cols_local)
                    combined_views_ref_phased_pd = pd.DataFrame(combined_view_ref_phased, columns=pd_cols_local)
                    for mf in mf_pairs_extract:
                        k = (a, mf)
                        views_target_dict[k] = combined_views_target_phased_pd[['a',mf]]
                        views_ref_dict[k] = combined_views_ref_phased_pd[['a',mf]]

            self.calc_utility_ranges(views_names=views_names, views_target_dict=views_target_dict, views_ref_dict=views_ref_dict, not_first_phase=True)

            self.top_5_largest_views = dict(sorted(self.views_utilities_ranges.items(), key = lambda x: x[1][1], reverse = True)[:5])
            print("Top 5 Largest Views = ", self.top_5_largest_views)
            lowest_lower_bound = sorted(self.top_5_largest_views.values(), key = lambda x: x[0], reverse = False)[:1]
            print("Lowest Lower Bound = ", lowest_lower_bound[0][0])


            print("Before pruning, Lenght = ", len(views_names))

            copy_names = views_names.copy()
            for name in copy_names:
                if self.views_utilities_ranges[name][1] < lowest_lower_bound[0][0]:
                    views_names.remove(name)
                    del self.views_utilities_ranges[name]
            print("After pruning, Lenght = ", len(views_names))

            self.cur.execute('drop table temp_table_target_{i}'.format(i=i))
            self.cur.execute('drop table temp_table_reference_{i}'.format(i=i))


        # print("The final Utility Ranges are ::")
        # for (k, v) in self.views_utilities_ranges.items():
        #     print("View name = ", k, ", Utility Range = ", v)

        self.create_visualisations()
        self.cur.close()




obj = SeeDB(["avg", "max", "min", "sum"], ["fnlwgt","capital_loss", "capital_gain", "hours_per_week", "age"], ["education", "income", "workclass", "occupation", "relationship", "race", "sex", "native_country"])
obj.main_method()
