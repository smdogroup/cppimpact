#pragma once
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

template <typename T, int nodes_per_element>
class Mesh {
 public:
  int num_elements, num_nodes, num_node_sets, num_fixed_nodes, num_slave_nodes;
  int *element_nodes, *node_set_starts, *node_set_indices, *fixed_nodes,
      *slave_nodes;
  T *xloc;
  std::vector<std::string> node_set_names;

  // Constructor
  Mesh()
      : num_elements(0),
        num_nodes(0),
        num_node_sets(0),
        num_fixed_nodes(0),
        num_slave_nodes(0),
        element_nodes(nullptr),
        node_set_starts(nullptr),
        node_set_indices(nullptr),
        xloc(nullptr),
        fixed_nodes(nullptr),
        slave_nodes(nullptr) {}

  // Destructor
  ~Mesh() {
    delete[] element_nodes;
    delete[] node_set_starts;
    delete[] node_set_indices;
    delete[] xloc;
  }

  std::string trim(const std::string &str) {
    // Include other whitespace characters like '\t', '\n', '\r', '\f', '\v'
    const std::string WHITESPACE = " \t\n\r\f\v";

    size_t first = str.find_first_not_of(WHITESPACE);
    if (first == std::string::npos) {
      return "";  // Return an empty string if only whitespace characters are
                  // found
    }

    size_t last = str.find_last_not_of(WHITESPACE);
    return str.substr(first, (last - first + 1));
  }

  // Main Parsing Logic
  int load_mesh(std::string filename) {
    // Data Structures
    struct Node {
      int index;
      double x, y, z;
    };

    struct Element {
      int index;
      std::vector<int> nodeIndices;
    };

    struct NodeSet {
      std::string name;
      std::vector<int> nodeIndices;
    };

    std::ifstream meshFile(filename);
    if (!meshFile.is_open()) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return 1;
    }

    std::string line;

    std::vector<Node> nodes;
    std::map<int, Element> elements;
    std::map<std::string, NodeSet> nodeSets;
    std::vector<std::string> node_set_names_temp;

    // Clear data in case function runs more than once
    delete[] element_nodes;
    element_nodes = nullptr;
    delete[] node_set_starts;
    node_set_starts = nullptr;
    delete[] node_set_indices;
    node_set_indices = nullptr;
    delete[] xloc;
    xloc = nullptr;

    bool inNodesSection = false, inElementsSection = false,
         inNodeSetsSection = false;
    std::string currentSetName;

    while (getline(meshFile, line)) {
      line = trim(line);
      if (line.empty() || line[0] == '*') {
        inNodesSection = line.find("*Node") != std::string::npos;
        inElementsSection = line.find("*Element") != std::string::npos;
        inNodeSetsSection = line.find("*Nset") != std::string::npos;

        if (inNodeSetsSection) {
          size_t namePos = line.find("Nset=");
          if (namePos != std::string::npos) {
            currentSetName = line.substr(namePos + 5);
            nodeSets[currentSetName] = NodeSet{currentSetName};
          }
        }

        continue;
      }

      if (inNodesSection) {
        std::istringstream iss(line);
        std::string indexStr;
        std::getline(iss, indexStr,
                     ',');  // Read up to the first comma to get the node index.
        int nodeIndex = std::stoi(indexStr);  // Convert index string to int.

        Node node;
        node.index = nodeIndex;

        std::string coordinateStr;
        std::getline(iss, coordinateStr,
                     ',');  // Read up to the next comma for the x coordinate.
        node.x = std::stod(coordinateStr);  // Convert to double.

        std::getline(iss, coordinateStr,
                     ',');  // Read up to the next comma for the y coordinate.
        node.y = std::stod(coordinateStr);  // Convert to double.

        std::getline(iss,
                     coordinateStr);  // Read the rest of the line for the z
                                      // coordinate (assuming no more commas).
        node.z = std::stod(coordinateStr);  // Convert to double.

        nodes.push_back(node);
      } else if (inElementsSection) {
        std::istringstream iss(line);
        Element element;
        if (!(iss >> element.index)) {  // Read and check the element's index.
          std::cerr << "Failed to read element index from line: " << line
                    << std::endl;
          continue;  // Skip to the next line if the element index can't be
                     // read.
        }

        // Read the rest of the line as a single string.
        std::string restOfLine;
        std::getline(iss, restOfLine);

        // Use another stringstream to parse the node indices from restOfLine.
        std::istringstream nodeStream(restOfLine);
        std::string
            nodeIndexStr;  // Use a string to temporarily hold each node index.

        while (std::getline(nodeStream, nodeIndexStr,
                            ',')) {     // Read up to the next comma.
          if (!nodeIndexStr.empty()) {  // Check if the string is not empty.
            std::istringstream indexStream(
                nodeIndexStr);  // Use another stringstream to convert string to
                                // int.
            int nodeIndex;
            if (indexStream >> nodeIndex) {  // Convert the string to an int.
              element.nodeIndices.push_back(nodeIndex);
            }
          }
        }
        elements[element.index] = element;
      } else if (inNodeSetsSection) {
        std::istringstream iss(
            line);  // Add a trailing comma to ensure the last token is parsed.
        std::string token;
        std::regex numberPattern("^\\d+$");

        while (std::getline(iss, token, ',')) {
          token = trim(token);

          if (!token.empty()) {
            int nodeIndex = std::stoi(token);
            nodeSets[currentSetName].nodeIndices.push_back(nodeIndex);
          }
        }
      }
    }

    meshFile.close();

    // Convert elements and nodeSets to flat structures
    num_elements = elements.size();
    int *elem_nodes = new int[nodes_per_element * num_elements];

    for (const auto &elem : elements) {
      for (int j = 0; j < nodes_per_element; j++) {
        elem_nodes[nodes_per_element * (elem.second.index - 1) + j] =
            elem.second.nodeIndices[j] - 1;
      }
    }

    num_nodes = nodes.size();
    T *x = new T[3 * num_nodes];

    for (const auto &node : nodes) {
      x[3 * (node.index - 1)] = node.x;
      x[3 * (node.index - 1) + 1] = node.y;
      x[3 * (node.index - 1) + 2] = node.z;
    }

    // Count the total number of indices across all node sets
    int totalNodeSetIndices = 0;
    for (const auto &nodeset : nodeSets) {
      totalNodeSetIndices += nodeset.second.nodeIndices.size();
    }

    int *flatNodeSetIndices = new int[totalNodeSetIndices];

    int *nodeSetStarts = new int[nodeSets.size() + 1];

    int currentIndex = 0;
    int nodeSetCount = 0;

    for (const auto &nodeset : nodeSets) {
      nodeSetStarts[nodeSetCount] = currentIndex;
      node_set_names_temp.push_back(nodeset.second.name);

      for (int nodeIndex : nodeset.second.nodeIndices) {
        flatNodeSetIndices[currentIndex++] = nodeIndex;
      }
      nodeSetCount++;
    }

    // The last number in nodeSetStarts is the total length of the array
    // To more easily iterate through the array
    if (!nodeSets.empty()) {
      nodeSetStarts[nodeSetCount] = totalNodeSetIndices;
    }

    // Account for the mesh indexing starting at 1

    if (nodeSets.size() == 0) {
      printf(
          "No slave node sets found. Walls will have no effect on "
          "simulation.\n");
    } else {
      for (int i = 0; i < nodeSetStarts[nodeSets.size()]; i++) {
        flatNodeSetIndices[i] = flatNodeSetIndices[i] - 1;
      }
    }

    num_node_sets = nodeSets.size();
    element_nodes = elem_nodes;
    xloc = x;
    node_set_starts = nodeSetStarts;
    node_set_indices = flatNodeSetIndices;
    node_set_names = node_set_names_temp;

    collect_fixed_nodes();
    collect_slave_nodes();

    return 0;
  }

 private:
  void collect_fixed_nodes() {
    num_fixed_nodes = 0;
    fixed_nodes = nullptr;

    // Regex looks for "fixed" in the set name, case insensitive
    std::regex fixed_regex("fixed", std::regex_constants::icase);

    for (int i = 0; i < num_node_sets; i++) {
      if (std::regex_search(
              node_set_names[i],
              fixed_regex)) {  // Check if name contains "fixed" using regex
        int start_idx = node_set_starts[i];
        int end_idx = node_set_starts[i + 1];

        // Resize fixed_nodes if necessary
        int *temp = new int[num_fixed_nodes + (end_idx - start_idx)];
        if (fixed_nodes != nullptr) {
          std::copy(fixed_nodes, fixed_nodes + num_fixed_nodes, temp);
          delete[] fixed_nodes;
        }
        fixed_nodes = temp;

        // Add node indices to fixed_nodes
        for (int j = start_idx; j < end_idx; ++j) {
          fixed_nodes[num_fixed_nodes++] = node_set_indices[j];
        }
      }
    }
  }

  // TODO: Handle files without a slave nodeset
  void collect_slave_nodes() {
    num_slave_nodes = 0;
    slave_nodes = nullptr;

    // Regex looks for "slave" in the set name, case insensitive
    std::regex slave_regex("slave", std::regex_constants::icase);

    for (int i = 0; i < num_node_sets; i++) {
      if (std::regex_search(
              node_set_names[i],
              slave_regex)) {  // Check if name contains "slave" using regex
        int start_idx = node_set_starts[i];
        int end_idx = node_set_starts[i + 1];

        // Resize slave_nodes if necessary
        int *temp = new int[num_slave_nodes + (end_idx - start_idx)];
        if (slave_nodes != nullptr) {
          std::copy(slave_nodes, slave_nodes + num_slave_nodes, temp);
          delete[] slave_nodes;
        }
        slave_nodes = temp;

        // Add node indices to slave_nodes
        for (int j = start_idx; j < end_idx; ++j) {
          slave_nodes[num_slave_nodes++] = node_set_indices[j];
        }
      }
    }
  }
};