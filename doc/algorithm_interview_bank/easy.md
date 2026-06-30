# Easy 高频题

## 1. [Two Sum](https://leetcode.cn/problems/two-sum/)

题目链接：[LeetCode 1 - Two Sum](https://leetcode.cn/problems/two-sum/)

完整题目：给定整数数组 `nums` 和整数 `target`，在数组中找出两个不同下标，使这两个数之和等于 `target`，返回这两个下标。题目保证通常只有一个有效答案，同一个元素不能使用两次。

题型：数组、哈希表。

思路：一边遍历一边记录已经见过的数。对当前数 `x`，检查 `target - x` 是否已经出现。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        seen = {}
        for i, x in enumerate(nums):
            need = target - x
            if need in seen:
                return [seen[need], i]
            seen[x] = i
        return []
```

## 2. [Contains Duplicate](https://leetcode.cn/problems/contains-duplicate/)

题目链接：[LeetCode 217 - Contains Duplicate](https://leetcode.cn/problems/contains-duplicate/)

完整题目：给定整数数组 `nums`，判断是否存在某个值在数组中至少出现两次；如果存在重复元素返回 `True`，否则返回 `False`。

题型：数组、哈希集合。

思路：如果去重后的长度小于原数组长度，说明存在重复元素。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
class Solution:
    def containsDuplicate(self, nums: list[int]) -> bool:
        return len(nums) != len(set(nums))
```

## 3. [Valid Anagram](https://leetcode.cn/problems/valid-anagram/)

题目链接：[LeetCode 242 - Valid Anagram](https://leetcode.cn/problems/valid-anagram/)

完整题目：给定两个字符串 `s` 和 `t`，判断 `t` 是否可以通过重新排列 `s` 中的全部字符得到，也就是两个字符串是否互为字母异位词。

题型：字符串、计数。

思路：两个字符串互为异位词，当且仅当每个字符出现次数相同。

复杂度：时间 `O(n)`，空间 `O(1)`，字符集固定时可视作常数空间。

```python
from collections import Counter


class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return Counter(s) == Counter(t)
```

## 4. [Valid Parentheses](https://leetcode.cn/problems/valid-parentheses/)

题目链接：[LeetCode 20 - Valid Parentheses](https://leetcode.cn/problems/valid-parentheses/)

完整题目：给定只包含 `(`、`)`、`[`、`]`、`{`、`}` 的字符串，判断括号是否有效：每个右括号必须匹配最近的同类型左括号，并且括号闭合顺序必须正确。

题型：栈。

思路：遇到左括号入栈；遇到右括号时，栈顶必须是对应的左括号。最后栈必须为空。

复杂度：时间 `O(n)`，空间 `O(n)`。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        pairs = {")": "(", "]": "[", "}": "{"}
        stack = []

        for ch in s:
            if ch in pairs:
                if not stack or stack.pop() != pairs[ch]:
                    return False
            else:
                stack.append(ch)

        return not stack
```

## 5. [Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

题目链接：[LeetCode 121 - Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

完整题目：给定数组 `prices`，其中 `prices[i]` 表示第 `i` 天股票价格。你最多只能买入一次并卖出一次，且买入必须发生在卖出之前，返回可以获得的最大利润；如果无法获利，返回 `0`。

题型：数组、一次遍历。

思路：遍历价格时维护历史最低买入价，用当前价格减去最低价更新最大利润。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        min_price = float("inf")
        best = 0

        for price in prices:
            min_price = min(min_price, price)
            best = max(best, price - min_price)

        return best
```

## 6. [Valid Palindrome](https://leetcode.cn/problems/valid-palindrome/)

题目链接：[LeetCode 125 - Valid Palindrome](https://leetcode.cn/problems/valid-palindrome/)

完整题目：给定字符串 `s`，忽略大小写并去掉所有非字母数字字符后，判断剩余字符串是否为回文串。

题型：字符串、双指针。

思路：左右指针跳过非字母数字字符，只比较有效字符的小写形式。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left, right = 0, len(s) - 1

        while left < right:
            while left < right and not s[left].isalnum():
                left += 1
            while left < right and not s[right].isalnum():
                right -= 1
            if s[left].lower() != s[right].lower():
                return False
            left += 1
            right -= 1

        return True
```

## 7. [Binary Search](https://leetcode.cn/problems/binary-search/)

题目链接：[LeetCode 704 - Binary Search](https://leetcode.cn/problems/binary-search/)

完整题目：给定一个升序排列且元素互不相同的整数数组 `nums` 和目标值 `target`，用 `O(log n)` 时间查找目标值下标；如果不存在，返回 `-1`。

题型：二分查找。

思路：在有序数组中维护闭区间 `[left, right]`。如果中点小于目标，答案只可能在右侧；反之在左侧。

复杂度：时间 `O(log n)`，空间 `O(1)`。

```python
class Solution:
    def search(self, nums: list[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

## 8. [Merge Two Sorted Lists](https://leetcode.cn/problems/merge-two-sorted-lists/)

题目链接：[LeetCode 21 - Merge Two Sorted Lists](https://leetcode.cn/problems/merge-two-sorted-lists/)

完整题目：给定两个升序链表 `list1` 和 `list2` 的头节点，将它们合并成一个新的升序链表，并返回合并后链表的头节点。

题型：链表、双指针。

思路：用 dummy 节点统一处理头节点。每次接上较小的节点，最后接上剩余链表。

复杂度：时间 `O(m + n)`，空间 `O(1)`。

```python
from typing import Optional


class Solution:
    def mergeTwoLists(
        self,
        list1: Optional[ListNode],
        list2: Optional[ListNode],
    ) -> Optional[ListNode]:
        dummy = ListNode()
        cur = dummy

        while list1 and list2:
            if list1.val <= list2.val:
                cur.next = list1
                list1 = list1.next
            else:
                cur.next = list2
                list2 = list2.next
            cur = cur.next

        cur.next = list1 or list2
        return dummy.next
```

## 9. [Reverse Linked List](https://leetcode.cn/problems/reverse-linked-list/)

题目链接：[LeetCode 206 - Reverse Linked List](https://leetcode.cn/problems/reverse-linked-list/)

完整题目：给定单链表头节点 `head`，反转整个链表，并返回反转后的头节点。

题型：链表、指针反转。

思路：维护 `prev` 和 `cur`，每次先保存 `cur.next`，再把当前节点指向前一个节点。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
from typing import Optional


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head

        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt

        return prev
```

## 10. [Linked List Cycle](https://leetcode.cn/problems/linked-list-cycle/)

题目链接：[LeetCode 141 - Linked List Cycle](https://leetcode.cn/problems/linked-list-cycle/)

完整题目：给定单链表头节点 `head`，判断链表中是否存在环。如果某个节点可以通过连续 `next` 指针再次回到自身，则说明链表有环。

题型：链表、快慢指针。

思路：慢指针每次走一步，快指针每次走两步。如果存在环，二者一定会相遇。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
from typing import Optional


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True

        return False
```

## 11. [Maximum Depth of Binary Tree](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

题目链接：[LeetCode 104 - Maximum Depth of Binary Tree](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

完整题目：给定二叉树根节点 `root`，返回树的最大深度，也就是从根节点到最远叶子节点路径上的节点数。

题型：二叉树、DFS。

思路：空树深度为 0；非空树深度为左右子树最大深度加 1。

复杂度：时间 `O(n)`，空间 `O(h)`，`h` 为树高。

```python
from typing import Optional


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

## 12. [Climbing Stairs](https://leetcode.cn/problems/climbing-stairs/)

题目链接：[LeetCode 70 - Climbing Stairs](https://leetcode.cn/problems/climbing-stairs/)

完整题目：你正在爬一段共有 `n` 阶的楼梯，每次可以爬 `1` 阶或 `2` 阶。返回爬到楼顶共有多少种不同方法。

题型：动态规划。

思路：到达第 `i` 阶的方法数等于到达 `i - 1` 和 `i - 2` 的方法数之和。只保留最近两个状态。

复杂度：时间 `O(n)`，空间 `O(1)`。

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        prev2, prev1 = 1, 1

        for _ in range(n):
            prev2, prev1 = prev1, prev1 + prev2

        return prev2
```

## 13. [Sqrt(x)](https://leetcode.cn/problems/sqrtx/)

题目链接：[LeetCode 69 - Sqrt(x)](https://leetcode.cn/problems/sqrtx/)

完整题目：给定一个非负整数 `x`，实现整数平方根函数，返回 `sqrt(x)` 向下取整后的整数。不能直接使用内置的开方函数或幂函数。

题型：二分查找。

思路：答案一定在 `[0, x]` 范围内。用二分查找最大的 `mid`，使得 `mid * mid <= x`。

复杂度：时间 `O(log x)`，空间 `O(1)`。

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left, right = 0, x

        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                left = mid + 1
            else:
                right = mid - 1

        return right
```
