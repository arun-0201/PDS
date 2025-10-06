#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Tree node structure
typedef struct TreeNode {
    int data;
    struct TreeNode* left;
    struct TreeNode* right;
} TreeNode;

// Function to create a new tree node
TreeNode* createNode(int data) {
    TreeNode* newNode = (TreeNode*)malloc(sizeof(TreeNode));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

// Preorder traversal function
void preorderTraversal(TreeNode* root, int rank) {
    if (root == NULL) return;

    printf("Process %d: Visiting node %d\n", rank, root->data);
    preorderTraversal(root->left, rank);
    preorderTraversal(root->right, rank);
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    TreeNode* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    if (rank == 0) {
        printf("=== Preorder Tree Traversal ===\n");
        printf("Tree structure: 1 -> (2 -> (4, 5), 3)\n");
        printf("Expected order: 1, 2, 4, 5, 3\n\n");
    }

    // Each process performs the traversal (for demonstration)
    // In a real distributed scenario, you might distribute the tree across processes
    preorderTraversal(root, rank);
    free(root->left->right);
    free(root->left->left);
    free(root->right);
    free(root->left);
    free(root);

    MPI_Finalize();
    return 0;
}
