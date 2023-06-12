#pragma once
#include <vector>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <omp.h>
#include <iostream>
/*
给出一个文法G，求出G的所有生成树的个数
使用openmp加速
文法G的定义：
G=(Vn,Vt,P,S)
Vn是非终结符集合
Vt是终结符集合
P是产生式集合
S是开始符号
产生式由两种，一种是非终结符生成两个非终结符，一种是非终结符生成一个终结符
一个简单实例
4
5
<0>::=<1><2>
<0>::=<2><3>
<1>::=<2><1>
<2>::=<3><3>
<3>::=<1><2>
3
<1>::=a
<2>::=b
<3>::=a
5
baaba
结果为2

cyk<char,int,string,unsigned> cyk;
cyk.SetVnNum(4);
cyk.AddNonterminalProduction(0,1,2);
...
cyk.AddTerminalProduction(1,'a');
...
cyk.GetNumProductionTree("baaba")结果为2
*/
namespace CYK
{
    using std::map;
    using std::pair;
    using std::set;
    using std::thread;
    using std::vector;
    template <typename VtType, typename VnType, typename TargetType, typename AnswerType>
    class CYK
    {
        // dp[l][r][symbol]代表l到r的子串能够推导出symbol的次数
        vector<vector<vector<AnswerType>>> dp;
        vector<pair<VnType, pair<VnType, VnType>>> production2;
        map<VtType, set<VnType>> vt_parent;
        map<pair<VnType, VnType>, set<VnType>> vn_parent;
        int vn_num;

    public:
        void SetVnNum(int vn_num)
        {
            this->vn_num = vn_num;
        }
        void AddNonterminalProduction(VnType parent, VnType child1, VnType child2)
        {
            production2.push_back({parent, {child1, child2}});
            vn_parent[{child1, child2}].insert(parent);
        }
        void AddTerminalProduction(VnType parent, VtType child)
        {
            vt_parent[child].insert(parent);
        }
        AnswerType GetNumProductionTree(TargetType targetString, int num_threads = std::thread::hardware_concurrency())
        {
            dp.resize(
                targetString.size(),
                vector<vector<AnswerType>>(
                    targetString.size(),
                    vector<AnswerType>(vn_num, 0)));
            omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(guided)
            for (int i = 0; i < targetString.size(); i++)
            {
                for (VnType symbol : vt_parent[targetString[i]])
                {
                    dp[i][i][symbol] = 1;
                }
            }
            for (int len = 2; len <= targetString.size(); len++)
            {
#pragma omp parallel for schedule(guided)
                for (int right = len - 1; right < targetString.size(); right++)
                {
                    int left = right - len + 1;
                    for (int k = left; k < right; k++)
                    {
                        for (int i = 0; i < production2.size(); i++)
                        {
                            VnType parent = production2[i].first, left_symbol = production2[i].second.first, right_symbol = production2[i].second.second;
                            dp[left][right][parent] += dp[left][k][left_symbol] * dp[k + 1][right][right_symbol];
                        }
                    }
                }
            }
            return dp[0][targetString.size() - 1][0];
        }
    };
}
