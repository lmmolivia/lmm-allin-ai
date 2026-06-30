# Hard 高频题

## 1. [Merge k Sorted Lists](https://leetcode.cn/problems/merge-k-sorted-lists/)

题目链接：[LeetCode 23 - Merge k Sorted Lists](https://leetcode.cn/problems/merge-k-sorted-lists/)

完整题目：给定 `k` 个升序链表的头节点数组 `lists`，将所有链表合并为一个升序链表，并返回合并后链表的头节点。

题型：链表、最小堆。

思路：把每条链表的头节点放入最小堆。每次弹出当前最小节点接到结果链表，再把它的下一个节点放入堆。

复杂度：时间 `O(N log k)`，空间 `O(k)`，`N` 为总节点数。

```python
from heapq import heappop, heappush
from typing import Optional


class Solution:
    def mergeKLists(self, lists: list[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []

        for i, node in enumerate(lists):
            if node:
                heappush(heap, (node.val, i, node))

        dummy = ListNode()
        cur = dummy

        while heap:
            _, i, node = heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heappush(heap, (node.next.val, i, node.next))

        return dummy.next
```

## 2. [Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)

题目链接：[LeetCode 42 - Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)

完整题目：给定非负整数数组 `height`，每个元素表示柱子高度。柱子宽度为 1，计算下雨后这些柱子之间最多能接多少雨水。

题型：数组、双指针。

思路：左右两侧最大高度中较小的一侧决定当前位置可接雨水量。移动较小侧，并更新该侧最大值。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def trap(self, height: list[int]) -> int:
        left, right = 0, len(height) - 1
        left_max = right_max = 0
        water = 0

        while left < right:
            if height[left] < height[right]:
                left_max = max(left_max, height[left])
                water += left_max - height[left]
                left += 1
            else:
                right_max = max(right_max, height[right])
                water += right_max - height[right]
                right -= 1

        return water
```

## 3. [Sliding Window Maximum](https://leetcode.cn/problems/sliding-window-maximum/)

题目链接：[LeetCode 239 - Sliding Window Maximum](https://leetcode.cn/problems/sliding-window-maximum/)

完整题目：给定数组 `nums` 和窗口大小 `k`，窗口从左到右每次移动一位，返回每个窗口中的最大值。

题型：滑动窗口、单调队列。

思路：队列里存下标，并保持对应值单调递减。窗口右移时删除过期下标，队首就是当前最大值。

复杂度：时间 `O(n)`，空间 `O(k)`。

```python
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        queue = deque()
        ans = []

        for i, x in enumerate(nums):
            while queue and queue[0] <= i - k:
                queue.popleft()
            while queue and nums[queue[-1]] <= x:
                queue.pop()

            queue.append(i)
            if i >= k - 1:
                ans.append(nums[queue[0]])

        return ans
```

## 4. [Minimum Window Substring](https://leetcode.cn/problems/minimum-window-substring/)

题目链接：[LeetCode 76 - Minimum Window Substring](https://leetcode.cn/problems/minimum-window-substring/)

完整题目：给定字符串 `s` 和 `t`，在 `s` 中找到包含 `t` 所有字符及其出现次数的最短子串；如果不存在，返回空字符串。

题型：字符串、滑动窗口。

思路：右指针扩张窗口直到覆盖 `t`，再移动左指针尽量收缩。用计数器记录还缺多少字符。

复杂度：时间 `O(len(s) + len(t))`，空间 `O(charset)`。

```python
from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = Counter(t)
        missing = len(t)
        left = 0
        best_left, best_right = 0, float("inf")

        for right, ch in enumerate(s):
            if need[ch] > 0:
                missing -= 1
            need[ch] -= 1

            while missing == 0:
                if right - left + 1 < best_right - best_left:
                    best_left, best_right = left, right + 1

                left_ch = s[left]
                need[left_ch] += 1
                if need[left_ch] > 0:
                    missing += 1
                left += 1

        return "" if best_right == float("inf") else s[best_left:best_right]
```

## 5. [Median of Two Sorted Arrays](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

题目链接：[LeetCode 4 - Median of Two Sorted Arrays](https://leetcode.cn/problems/median-of-two-sorted-arrays/)

完整题目：给定两个升序数组 `nums1` 和 `nums2`，返回合并后数组的中位数，要求整体时间复杂度达到 `O(log(m + n))`。

题型：二分查找。

思路：在较短数组上二分切分位置，使左半部分总长度固定，且两个数组左半最大值不大于右半最小值。

复杂度：时间 `O(log min(m, n))`，空间 `O(1)`。

```python
class Solution:
    def findMedianSortedArrays(self, nums1: list[int], nums2: list[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)

        m, n = len(nums1), len(nums2)
        total = m + n
        half = (total + 1) // 2
        left, right = 0, m

        while left <= right:
            i = (left + right) // 2
            j = half - i

            a_left = nums1[i - 1] if i > 0 else float("-inf")
            a_right = nums1[i] if i < m else float("inf")
            b_left = nums2[j - 1] if j > 0 else float("-inf")
            b_right = nums2[j] if j < n else float("inf")

            if a_left <= b_right and b_left <= a_right:
                if total % 2 == 1:
                    return float(max(a_left, b_left))
                return (max(a_left, b_left) + min(a_right, b_right)) / 2
            if a_left > b_right:
                right = i - 1
            else:
                left = i + 1

        return 0.0
```

## 6. [Word Ladder](https://leetcode.cn/problems/word-ladder/)

题目链接：[LeetCode 127 - Word Ladder](https://leetcode.cn/problems/word-ladder/)

完整题目：给定 `beginWord`、`endWord` 和单词表 `wordList`。每次只能改变一个字母，且中间单词必须在单词表中，返回从起点变到终点所需的最短转换序列长度；不可达返回 `0`。

题型：图、BFS。

思路：每个单词看作节点，相差一个字符的单词之间有边。BFS 第一次到达 `endWord` 的层数就是最短转换长度。

复杂度：时间 `O(n * L * 26)`，空间 `O(n)`，`L` 为单词长度。

```python
from collections import deque


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: list[str]) -> int:
        words = set(wordList)
        if endWord not in words:
            return 0

        words.discard(beginWord)
        queue = deque([(beginWord, 1)])

        while queue:
            word, dist = queue.popleft()
            if word == endWord:
                return dist

            for i in range(len(word)):
                for code in range(ord("a"), ord("z") + 1):
                    nxt = word[:i] + chr(code) + word[i + 1:]
                    if nxt in words:
                        words.remove(nxt)
                        queue.append((nxt, dist + 1))

        return 0
```

## 7. [Largest Rectangle in Histogram](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

题目链接：[LeetCode 84 - Largest Rectangle in Histogram](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

完整题目：给定数组 `heights`，每个元素表示柱状图中宽度为 1 的柱子高度，返回柱状图中可以形成的最大矩形面积。

题型：单调栈。

思路：维护高度递增的栈。遇到更矮的柱子时，弹出栈顶作为矩形高度，当前下标和新栈顶决定宽度。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
class Solution:
    def largestRectangleArea(self, heights: list[int]) -> int:
        stack = []
        best = 0

        for i, h in enumerate(heights + [0]):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                left = stack[-1] if stack else -1
                width = i - left - 1
                best = max(best, height * width)
            stack.append(i)

        return best
```

## 8. [Edit Distance](https://leetcode.cn/problems/edit-distance/)

题目链接：[LeetCode 72 - Edit Distance](https://leetcode.cn/problems/edit-distance/)

完整题目：给定两个字符串 `word1` 和 `word2`，可以插入、删除或替换一个字符，返回把 `word1` 转换成 `word2` 所需的最少操作数。

题型：动态规划。

思路：`dp[i][j]` 表示 `word1[:i]` 转成 `word2[:j]` 的最少操作数。只需要上一行即可滚动优化。

复杂度：时间 `O(mn)`，空间 `O(n)`。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        prev = list(range(n + 1))

        for i in range(1, m + 1):
            cur = [i] + [0] * n
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    cur[j] = prev[j - 1]
                else:
                    cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
            prev = cur

        return prev[n]
```

## 9. [Serialize and Deserialize Binary Tree](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

题目链接：[LeetCode 297 - Serialize and Deserialize Binary Tree](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

完整题目：设计二叉树的序列化和反序列化方法。序列化要把树编码成字符串，反序列化要能从字符串恢复出原来的树结构。

题型：二叉树、BFS、编码设计。

思路：序列化时层序遍历，把空节点写成 `#`。反序列化时按顺序给每个队列节点恢复左右孩子。

复杂度：序列化和反序列化都是时间 `O(n)`，空间 `O(n)`。

```python
from collections import deque
from typing import Optional


class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        if not root:
            return ""

        vals = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                vals.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                vals.append("#")

        while vals and vals[-1] == "#":
            vals.pop()

        return ",".join(vals)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        if not data:
            return None

        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        queue = deque([root])
        i = 1

        while queue and i < len(vals):
            node = queue.popleft()

            if vals[i] != "#":
                node.left = TreeNode(int(vals[i]))
                queue.append(node.left)
            i += 1

            if i < len(vals) and vals[i] != "#":
                node.right = TreeNode(int(vals[i]))
                queue.append(node.right)
            i += 1

        return root
```
