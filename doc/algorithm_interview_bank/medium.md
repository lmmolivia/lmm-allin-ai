# Medium 高频题

## 1. [Group Anagrams](https://leetcode.cn/problems/group-anagrams/)

题目链接：[LeetCode 49 - Group Anagrams](https://leetcode.cn/problems/group-anagrams/)

完整题目：给定字符串数组 `strs`，把互为字母异位词的字符串分到同一组中。每组内部顺序和组之间顺序通常不作要求。

题型：字符串、哈希表、计数。

思路：异位词的每个字母出现次数完全相同。把每个字符串转成长度为 26 的计数 tuple，作为哈希表 key；key 相同的字符串放到同一组。

复杂度：时间 `O(nk)`，空间 `O(nk)`，`k` 为单词平均长度。相比排序 key 的 `O(n k log k)` 更优。

```python
from collections import defaultdict


class Solution:
    def groupAnagrams(self, strs: list[str]) -> list[list[str]]:
        groups = defaultdict(list)

        for word in strs:
            counts = [0] * 26
            for ch in word:
                counts[ord(ch) - ord("a")] += 1
            groups[tuple(counts)].append(word)

        return list(groups.values())
```

## 2. [Longest Substring Without Repeating Characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

题目链接：[LeetCode 3 - Longest Substring Without Repeating Characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

完整题目：给定字符串 `s`，返回其中不含重复字符的最长连续子串长度。注意是子串，不是子序列。

题型：字符串、滑动窗口。

思路：维护无重复窗口 `[left, right]`。如果字符上次出现位置在窗口内，就把 `left` 移到它的下一位。

复杂度：时间 `O(n)`，空间 `O(min(n, charset))`。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        last = {}
        left = 0
        best = 0

        for right, ch in enumerate(s):
            if ch in last and last[ch] >= left:
                left = last[ch] + 1
            last[ch] = right
            best = max(best, right - left + 1)

        return best
```

## 3. [3Sum](https://leetcode.cn/problems/3sum/)

题目链接：[LeetCode 15 - 3Sum](https://leetcode.cn/problems/3sum/)

完整题目：给定整数数组 `nums`，找出所有不重复的三元组 `[nums[i], nums[j], nums[k]]`，满足三个下标互不相同且三数之和为 `0`。

题型：排序、双指针。

思路：先排序，枚举第一个数，再在右侧用双指针找两数和。跳过重复值避免重复三元组。

复杂度：时间 `O(n^2)`，空间 `O(1)`，不计输出。

```python
class Solution:
    def threeSum(self, nums: list[int]) -> list[list[int]]:
        nums.sort()
        ans = []

        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            left, right = i + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1

        return ans
```

## 4. [Container With Most Water](https://leetcode.cn/problems/container-with-most-water/)

题目链接：[LeetCode 11 - Container With Most Water](https://leetcode.cn/problems/container-with-most-water/)

完整题目：给定数组 `height`，第 `i` 个元素表示坐标 `i` 处竖线高度。选择两条竖线与 x 轴组成容器，返回能盛最多水的面积。

题型：数组、双指针。

思路：面积由较短边决定。左右指针从两端开始，每次移动较短的边，才可能得到更高的边。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def maxArea(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        best = 0

        while left < right:
            best = max(best, (right - left) * min(height[left], height[right]))
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1

        return best
```

## 5. [Product of Array Except Self](https://leetcode.cn/problems/product-of-array-except-self/)

题目链接：[LeetCode 238 - Product of Array Except Self](https://leetcode.cn/problems/product-of-array-except-self/)

完整题目：给定整数数组 `nums`，返回数组 `answer`，其中 `answer[i]` 等于除 `nums[i]` 之外所有元素的乘积。要求不使用除法，并尽量在 `O(n)` 时间完成。

题型：数组、前后缀积。

思路：先把每个位置左侧乘积写入答案，再从右向左乘上右侧乘积。不使用除法。

复杂度：时间 `O(n)`，额外空间 `O(1)`，不计输出数组。

```python
class Solution:
    def productExceptSelf(self, nums: list[int]) -> list[int]:
        ans = [1] * len(nums)

        prefix = 1
        for i, x in enumerate(nums):
            ans[i] = prefix
            prefix *= x

        suffix = 1
        for i in range(len(nums) - 1, -1, -1):
            ans[i] *= suffix
            suffix *= nums[i]

        return ans
```

## 6. [Top K Frequent Elements](https://leetcode.cn/problems/top-k-frequent-elements/)

题目链接：[LeetCode 347 - Top K Frequent Elements](https://leetcode.cn/problems/top-k-frequent-elements/)

完整题目：给定整数数组 `nums` 和整数 `k`，返回出现频率最高的 `k` 个元素。答案顺序通常不作要求。

题型：哈希表、桶排序。

思路：先统计每个元素的频次。频次最大不会超过 `n`，所以可以建立 `n + 1` 个桶，`bucket[f]` 存所有出现 `f` 次的元素。再从高频桶往低频桶收集，直到拿到 `k` 个元素。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
from collections import Counter


class Solution:
    def topKFrequent(self, nums: list[int], k: int) -> list[int]:
        counts = Counter(nums)
        buckets = [[] for _ in range(len(nums) + 1)]

        for num, freq in counts.items():
            buckets[freq].append(num)

        ans = []
        for freq in range(len(buckets) - 1, 0, -1):
            for num in buckets[freq]:
                ans.append(num)
                if len(ans) == k:
                    return ans

        return ans
```

## 7. [Search in Rotated Sorted Array](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

题目链接：[LeetCode 33 - Search in Rotated Sorted Array](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

完整题目：一个原本严格升序的数组在未知位置发生旋转。给定旋转后的数组 `nums` 和目标值 `target`，要求在 `O(log n)` 时间内返回目标下标；不存在则返回 `-1`。

题型：二分查找。

思路：旋转数组每次至少有一半是有序的。先判断哪一半有序，再判断目标是否落在这一半。

复杂度：时间 `O(log n)`，空间 `O(1)`。

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1
```

## 8. [Find Minimum in Rotated Sorted Array](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

题目链接：[LeetCode 153 - Find Minimum in Rotated Sorted Array](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

完整题目：给定一个由升序数组旋转得到的数组 `nums`，数组元素互不相同，返回其中的最小元素，要求使用 `O(log n)` 时间。

题型：二分查找。

思路：如果 `nums[mid] > nums[right]`，最小值在右半边；否则最小值在左半边含 `mid`。

复杂度：时间 `O(log n)`，空间 `O(1)`。

```python
class Solution:
    def findMin(self, nums: list[int]) -> int:
        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        return nums[left]
```

## 9. [Add Two Numbers](https://leetcode.cn/problems/add-two-numbers/)

题目链接：[LeetCode 2 - Add Two Numbers](https://leetcode.cn/problems/add-two-numbers/)

完整题目：给定两个非空链表 `l1` 和 `l2`，分别表示两个非负整数，数字按逆序存储，每个节点存一位。返回两数相加后的链表，结果也按逆序存储。

题型：链表、模拟。

思路：同时遍历两个链表和进位 `carry`。每一位求和后取个位作为新节点，十位作为进位。

复杂度：时间 `O(max(m, n))`，空间 `O(max(m, n))`。

```python
from typing import Optional


class Solution:
    def addTwoNumbers(
        self,
        l1: Optional[ListNode],
        l2: Optional[ListNode],
    ) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy
        carry = 0

        while l1 or l2 or carry:
            total = carry
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next

            carry, digit = divmod(total, 10)
            cur.next = ListNode(digit)
            cur = cur.next

        return dummy.next
```

## 10. [Binary Tree Level Order Traversal](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

题目链接：[LeetCode 102 - Binary Tree Level Order Traversal](https://leetcode.cn/problems/binary-tree-level-order-traversal/)

完整题目：给定二叉树根节点 `root`，按层从左到右遍历节点，返回每一层节点值组成的二维数组。

题型：二叉树、BFS。

思路：队列按层遍历。每轮固定当前层节点数量，把下一层节点加入队列。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
from collections import deque
from typing import Optional


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> list[list[int]]:
        if not root:
            return []

        ans = []
        queue = deque([root])

        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(level)

        return ans
```

## 11. [Validate Binary Search Tree](https://leetcode.cn/problems/validate-binary-search-tree/)

题目链接：[LeetCode 98 - Validate Binary Search Tree](https://leetcode.cn/problems/validate-binary-search-tree/)

完整题目：给定二叉树根节点 `root`，判断它是否是一棵有效的二叉搜索树：任意节点的左子树所有值都小于该节点，右子树所有值都大于该节点，并且左右子树也必须满足同样规则。

题型：二叉搜索树、DFS。

思路：递归时传入当前节点允许的开区间 `(low, high)`。左子树上界变为当前值，右子树下界变为当前值。

复杂度：时间 `O(n)`，空间 `O(h)`。

```python
from typing import Optional


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def dfs(node, low, high):
            if not node:
                return True
            if not (low < node.val < high):
                return False
            return dfs(node.left, low, node.val) and dfs(node.right, node.val, high)

        return dfs(root, float("-inf"), float("inf"))
```

## 12. [Lowest Common Ancestor of a BST](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

题目链接：[LeetCode 235 - Lowest Common Ancestor of a BST](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

完整题目：给定一棵二叉搜索树以及其中两个节点 `p` 和 `q`，返回它们的最近公共祖先，也就是同时拥有 `p`、`q` 作为后代且深度最大的节点。

题型：二叉搜索树。

思路：利用 BST 性质。如果 `p`、`q` 都小于当前节点，往左；都大于当前节点，往右；否则当前节点就是最近公共祖先。

复杂度：时间 `O(h)`，空间 `O(1)`。

```python
from typing import Optional


class Solution:
    def lowestCommonAncestor(
        self,
        root: Optional[TreeNode],
        p: TreeNode,
        q: TreeNode,
    ) -> Optional[TreeNode]:
        cur = root

        while cur:
            if p.val < cur.val and q.val < cur.val:
                cur = cur.left
            elif p.val > cur.val and q.val > cur.val:
                cur = cur.right
            else:
                return cur

        return None
```

## 13. [Kth Smallest Element in a BST](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

题目链接：[LeetCode 230 - Kth Smallest Element in a BST](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)

完整题目：给定二叉搜索树根节点 `root` 和整数 `k`，返回树中第 `k` 小的节点值。

题型：二叉搜索树、中序遍历。

思路：BST 中序遍历是升序。迭代中序遍历，每弹出一个节点就计数。

复杂度：时间 `O(h + k)`，空间 `O(h)`。

```python
from typing import Optional


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        cur = root

        while cur or stack:
            while cur:
                stack.append(cur)
                cur = cur.left

            cur = stack.pop()
            k -= 1
            if k == 0:
                return cur.val
            cur = cur.right

        raise ValueError("k is larger than the number of nodes")
```

## 14. [Number of Islands](https://leetcode.cn/problems/number-of-islands/)

题目链接：[LeetCode 200 - Number of Islands](https://leetcode.cn/problems/number-of-islands/)

完整题目：给定由字符 `1` 和 `0` 组成的二维网格，`1` 表示陆地，`0` 表示水。上下左右相邻的陆地连成岛屿，返回岛屿数量。

题型：图、DFS/BFS、网格。

思路：遍历网格，每遇到一个未访问陆地就计数，并用 DFS 把相连陆地全部标记为水。

复杂度：时间 `O(mn)`，空间 `O(mn)`，递归栈最坏覆盖整个网格。

```python
class Solution:
    def numIslands(self, grid: list[list[str]]) -> int:
        if not grid:
            return 0

        rows, cols = len(grid), len(grid[0])

        def dfs(r: int, c: int) -> None:
            if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != "1":
                return
            grid[r][c] = "0"
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)

        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1":
                    count += 1
                    dfs(r, c)

        return count
```

## 15. [Clone Graph](https://leetcode.cn/problems/clone-graph/)

题目链接：[LeetCode 133 - Clone Graph](https://leetcode.cn/problems/clone-graph/)

完整题目：给定一个无向连通图中某个节点 `node`，图节点包含 `val` 和 `neighbors`。返回整张图的深拷贝，使新图结构和值相同但节点对象全新。

题型：图、DFS、哈希表。

思路：用哈希表记录原节点到克隆节点的映射。DFS 遇到旧节点直接返回，避免环导致无限递归。

复杂度：时间 `O(V + E)`，空间 `O(V)`。

```python
from typing import Optional


class Solution:
    def cloneGraph(self, node: Optional["Node"]) -> Optional["Node"]:
        if not node:
            return None

        clones = {}

        def dfs(cur):
            if cur in clones:
                return clones[cur]

            copy = Node(cur.val)
            clones[cur] = copy
            for nei in cur.neighbors:
                copy.neighbors.append(dfs(nei))
            return copy

        return dfs(node)
```

## 16. [Course Schedule](https://leetcode.cn/problems/course-schedule/)

题目链接：[LeetCode 207 - Course Schedule](https://leetcode.cn/problems/course-schedule/)

完整题目：给定课程数量 `numCourses` 和先修关系 `prerequisites`，其中 `[a, b]` 表示学课程 `a` 前必须先学 `b`。判断是否可以完成所有课程。

题型：有向图、拓扑排序。

思路：课程依赖构成有向图。把入度为 0 的课程入队，不断删除它们的出边。最后能学习的课程数等于总数，说明无环。

复杂度：时间 `O(V + E)`，空间 `O(V + E)`。

```python
from collections import deque


class Solution:
    def canFinish(self, numCourses: int, prerequisites: list[list[int]]) -> bool:
        graph = [[] for _ in range(numCourses)]
        indegree = [0] * numCourses

        for course, pre in prerequisites:
            graph[pre].append(course)
            indegree[course] += 1

        queue = deque(i for i, deg in enumerate(indegree) if deg == 0)
        taken = 0

        while queue:
            cur = queue.popleft()
            taken += 1
            for nxt in graph[cur]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        return taken == numCourses
```

## 17. [House Robber](https://leetcode.cn/problems/house-robber/)

题目链接：[LeetCode 198 - House Robber](https://leetcode.cn/problems/house-robber/)

完整题目：给定数组 `nums`，其中 `nums[i]` 表示第 `i` 间房的钱。相邻房屋不能在同一晚被偷，返回在不触发警报的情况下最多能偷到多少钱。

题型：动态规划。

思路：到每一间房时只有两种选择：偷当前房，则不能偷上一间；不偷当前房，则保持上一间的最优值。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2 = 0
        prev1 = 0

        for money in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + money)

        return prev1
```

## 18. [Coin Change](https://leetcode.cn/problems/coin-change/)

题目链接：[LeetCode 322 - Coin Change](https://leetcode.cn/problems/coin-change/)

完整题目：给定硬币面额数组 `coins` 和总金额 `amount`，每种硬币数量无限，返回凑成该金额所需的最少硬币数；如果无法凑成，返回 `-1`。

题型：动态规划、完全背包。

思路：`dp[x]` 表示凑出金额 `x` 的最少硬币数。对每个金额枚举硬币，取最小值。

复杂度：时间 `O(amount * len(coins))`，空间 `O(amount)`。

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        inf = amount + 1
        dp = [0] + [inf] * amount

        for total in range(1, amount + 1):
            for coin in coins:
                if total >= coin:
                    dp[total] = min(dp[total], dp[total - coin] + 1)

        return -1 if dp[amount] == inf else dp[amount]
```

## 19. [Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

题目链接：[LeetCode 300 - Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

完整题目：给定整数数组 `nums`，返回最长严格递增子序列的长度。子序列可以删除若干元素但不能改变剩余元素相对顺序。

题型：动态规划、贪心、二分查找。

思路：维护 `tails[i]` 表示长度为 `i + 1` 的递增子序列的最小结尾。每个数用二分放到第一个大于等于它的位置。

复杂度：时间 `O(n log n)`，空间 `O(n)`。

```python
from bisect import bisect_left


class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        tails = []

        for x in nums:
            i = bisect_left(tails, x)
            if i == len(tails):
                tails.append(x)
            else:
                tails[i] = x

        return len(tails)
```

## 20. [Word Break](https://leetcode.cn/problems/word-break/)

题目链接：[LeetCode 139 - Word Break](https://leetcode.cn/problems/word-break/)

完整题目：给定字符串 `s` 和单词字典 `wordDict`，判断 `s` 是否可以被拆分成一个或多个字典中的单词。字典里的单词可以重复使用。

题型：字符串、动态规划、Trie。

思路：`dp[i]` 表示 `s[:i]` 能否被拆分。把字典建成 Trie，从每个可达位置 `i` 出发向后匹配单词；匹配到一个单词结尾，就把对应位置标记为可达。这样避免枚举所有 `j` 和频繁生成无效子串。

复杂度：建 Trie 时间 `O(total)`，其中 `total` 是字典总字符数；匹配时间 `O(nL)`，`L` 为字典中最长单词长度；空间 `O(total + n)`。

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        end = "#"
        trie = {}
        max_len = 0

        for word in wordDict:
            max_len = max(max_len, len(word))
            node = trie
            for ch in word:
                node = node.setdefault(ch, {})
            node[end] = True

        dp = [False] * (len(s) + 1)
        dp[0] = True

        for i in range(len(s)):
            if not dp[i]:
                continue

            node = trie
            stop = min(len(s), i + max_len)
            for j in range(i, stop):
                ch = s[j]
                if ch not in node:
                    break
                node = node[ch]
                if end in node:
                    dp[j + 1] = True

        return dp[-1]
```

## 21. [Random Pick with Weight](https://leetcode.cn/problems/random-pick-with-weight/)

题目链接：[LeetCode 528 - Random Pick with Weight](https://leetcode.cn/problems/random-pick-with-weight/)

完整题目：给定一个正整数权重数组 `w`，实现 `pickIndex()`，随机返回一个下标 `i`，并且下标 `i` 被返回的概率应当与 `w[i]` 成正比。

题型：随机采样、Alias Method。

思路：用 Alias Method 预处理两张表：`prob[i]` 表示直接返回 `i` 的概率阈值，`alias[i]` 表示没命中阈值时跳转到的备用下标。每次先等概率随机选一个桶，再根据桶内概率决定返回原下标还是别名下标。

复杂度：初始化时间 `O(n)`，空间 `O(n)`；每次采样时间 `O(1)`，空间 `O(1)`。

```python
import random


class Solution:
    def __init__(self, w: list[int]):
        n = len(w)
        total = sum(w)
        scaled = [weight * n / total for weight in w]
        small = []
        large = []

        self.prob = [0.0] * n
        self.alias = [0] * n

        for i, p in enumerate(scaled):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            s = small.pop()
            l = large.pop()

            self.prob[s] = scaled[s]
            self.alias[s] = l
            scaled[l] -= 1 - scaled[s]

            if scaled[l] < 1:
                small.append(l)
            else:
                large.append(l)

        for i in small + large:
            self.prob[i] = 1.0

    def pickIndex(self) -> int:
        i = random.randrange(len(self.prob))
        if random.random() < self.prob[i]:
            return i
        return self.alias[i]
```

## 22. [Random Pick Index](https://leetcode.cn/problems/random-pick-index/)

题目链接：[LeetCode 398 - Random Pick Index](https://leetcode.cn/problems/random-pick-index/)

完整题目：给定可能包含重复元素的整数数组 `nums`，实现 `pick(target)`，从所有满足 `nums[i] == target` 的下标中等概率随机返回一个。题目通常会多次调用 `pick`。

题型：哈希表、随机采样。

思路：题目会多次调用 `pick`，所以初始化时把每个值出现的所有下标存进哈希表。调用 `pick(target)` 时，直接从对应下标列表中等概率随机选一个。

复杂度：初始化时间 `O(n)`，空间 `O(n)`；每次 `pick` 时间 `O(1)`，空间 `O(1)`。

```python
import random
from collections import defaultdict


class Solution:
    def __init__(self, nums: list[int]):
        self.indices = defaultdict(list)
        for i, x in enumerate(nums):
            self.indices[x].append(i)

    def pick(self, target: int) -> int:
        return random.choice(self.indices[target])
```

## 23. [Implement Rand10() Using Rand7()](https://leetcode.cn/problems/implement-rand10-using-rand7/)

题目链接：[LeetCode 470 - Implement Rand10() Using Rand7()](https://leetcode.cn/problems/implement-rand10-using-rand7/)

完整题目：已知系统提供 `rand7()`，它能等概率返回 `1` 到 `7` 的整数。要求只使用 `rand7()` 实现 `rand10()`，使其等概率返回 `1` 到 `10` 的整数。

题型：拒绝采样、随机数生成。

思路：调用两次 `rand7()` 可以构造出 `1` 到 `49` 的等概率整数。因为 `40` 可以被 `10` 整除，所以只接受 `1..40`，把它映射到 `1..10`；如果落到 `41..49`，就丢弃并重试。

复杂度：期望时间 `O(1)`，空间 `O(1)`。

```python
class Solution:
    def rand10(self) -> int:
        while True:
            num = (rand7() - 1) * 7 + rand7()
            if num <= 40:
                return 1 + (num - 1) % 10
```
