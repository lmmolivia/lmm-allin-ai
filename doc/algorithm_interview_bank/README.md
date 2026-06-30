# 面试高频算法题库

这是一份面试前快速复习版题库，按 LeetCode 难度分为 Easy、Medium、Hard。每题包含官方链接、完整题目、题型、解题思路、复杂度和 Python 实现。

## 题目选择依据

公开资料通常给的是精选清单，不是实时公司面试频次数据库。因此这里把“高频”定义为：在多个主流面试题单中反复出现，并且覆盖数组、哈希表、双指针、滑动窗口、链表、树、图、动态规划、堆、单调栈等核心模式。

参考来源：

- [LeetCode 精选 TOP 面试题](https://leetcode.cn/problem-list/2ckc81c/)
- [Tech Interview Handbook](https://github.com/yangshun/tech-interview-handbook)，其 README 明确包含 Blind 75、Grind 75、best practice questions 和 algorithm cheatsheets。
- [NeetCode 150](https://neetcode.io/practice?tab=neetcode150)

更新时间：2026-06-25。

## 文件结构

- [easy.md](./easy.md)：13 题，基础高频题，优先保证一遍 AC。
- [medium.md](./medium.md)：23 题，面试主战场，覆盖最常考模式和随机采样题。
- [hard.md](./hard.md)：9 题，高频 Hard，不要求全背，但要会讲核心套路。

## 临场复习顺序

1. 先刷 Easy：Two Sum、Valid Parentheses、Best Time、Reverse Linked List、Binary Search、Sqrt(x)、Tree Depth。
2. 再刷 Medium：3Sum、Product Except Self、Rotated Binary Search、Number of Islands、Course Schedule、Coin Change、LIS、Random Pick with Weight。
3. 最后看 Hard：Merge k Lists、Trapping Rain Water、Sliding Window Maximum、Minimum Window Substring、Median of Two Sorted Arrays。

## 面试答题模板

1. 先说暴力解，明确为什么慢。
2. 再说优化关键：哈希、排序、双指针、单调结构、BFS/DFS、DP 状态。
3. 写代码时先处理空输入，再处理主逻辑。
4. 写完主动报复杂度。

## Python 约定

- 链表和树相关题默认使用 LeetCode 已提供的 `ListNode`、`TreeNode`、`Node` 定义。
- 代码以可读性和面试手写稳定性为优先，少量题使用 Python 标准库如 `collections`、`heapq`、`bisect`。
