#include <stdio.h>
#include <stdlib.h>

// Node structure
struct Node {
    int data;
    struct Node* next;
};

// Function to create a new node
struct Node* createNode(int data) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

// Function to reverse a linked list
struct Node* reverseList(struct Node* head) {
    struct Node* prev = NULL;
    struct Node* curr = head;
    struct Node* next;
    while (curr != NULL) {
        next = curr->next;
        curr->next = prev;
        prev = curr;
        curr = next;
    }
    return prev;
}

// Function to check if linked list is palindrome
int isPalindrome(struct Node* head) {
    if (head == NULL || head->next == NULL)
        return 1; // Empty or single node is palindrome

    // Step 1: Find middle using slow and fast pointers
    struct Node* slow = head;
    struct Node* fast = head;
    while (fast->next != NULL && fast->next->next != NULL) {
        slow = slow->next;
        fast = fast->next->next;
    }

    // Step 2: Reverse second half
    struct Node* secondHalf = reverseList(slow->next);

    // Step 3: Compare first half and second half
    struct Node* firstHalf = head;
    struct Node* tempSecond = secondHalf;
    int result = 1;
    while (tempSecond != NULL) {
        if (firstHalf->data != tempSecond->data) {
            result = 0; // Not a palindrome
            break;
        }
        firstHalf = firstHalf->next;
        tempSecond = tempSecond->next;
    }

    // Step 4 (Optional): Restore original list
    slow->next = reverseList(secondHalf);

    return result;
}

// Function to print linked list
void printList(struct Node* head) {
    while (head != NULL) {
        printf("%d -> ", head->data);
        head = head->next;
    }
    printf("NULL\n");
}

// Example usage
int main() {
    struct Node* head = createNode(1);
    head->next = createNode(2);
    head->next->next = createNode(3);
    head->next->next->next = createNode(2);
    head->next->next->next->next = createNode(1);

    printf("Linked list: ");
    printList(head);

    if (isPalindrome(head))
        printf("The linked list is a palindrome.\n");
    else
        printf("The linked list is not a palindrome.\n");

    return 0;
}
