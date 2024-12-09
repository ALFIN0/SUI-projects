#include "search-strategies.h"
#include <queue>
#include <set>
#include <stack>
#include <map>

std::vector<SearchAction> BreadthFirstSearch::solve(const SearchState &init_state) {
	std::queue<std::pair<SearchState, std::vector<SearchAction>>> frontier;
	std::set<SearchState> visited;

	frontier.push({init_state, {}});
	visited.insert(init_state);

	while (!frontier.empty()) {
		auto [current_state, actions] = frontier.front();
		frontier.pop();

		if (current_state.isFinal()) {
			return actions;
		}

		for (const auto &action : current_state.actions()) {
			SearchState new_state = action.execute(current_state);

			if (visited.find(new_state) == visited.end()) {
				visited.insert(new_state);
				auto new_actions = actions;
				new_actions.push_back(action);
				frontier.push({new_state, new_actions});
			}
		}
	}

	return {};
}

std::vector<SearchAction> DepthFirstSearch::solve(const SearchState &init_state) {
	std::stack<std::pair<SearchState, std::vector<SearchAction>>> frontier;
    frontier.push({init_state, {}});

    while (!frontier.empty()) {
        auto [current_state, actions] = frontier.top();
        frontier.pop();

        if (current_state.isFinal()) {
            return actions;
        }

        if (actions.size() < static_cast<size_t>(depth_limit_)) {
            for (const auto &action : current_state.actions()) {
                SearchState new_state = action.execute(current_state);
                auto new_actions = actions;
                new_actions.push_back(action);
                frontier.push({new_state, new_actions});
            }
        }
    }
    return {};
}

double StudentHeuristic::distanceLowerBound(const GameState &state) const {
    return 0;
}

struct Node {
        SearchState state;
        std::vector<SearchAction> actions;
        double cost;
        bool operator>(const Node &other) const {
            return cost > other.cost;
        }
    };

std::vector<SearchAction> AStarSearch::solve(const SearchState &init_state) {

    std::priority_queue<Node, std::vector<Node>, std::greater<>> frontier;
    std::map<const SearchState*, double> visited;

    // Put the initial state on the frontier
    frontier.push({init_state, {}, 0.0});
    visited[&init_state] = 0.0;

    while (!frontier.empty()) {
        auto current_node = frontier.top();
        frontier.pop();

        // If it's a goal state
        if (current_node.state.isFinal()) {
            return current_node.actions;
        }

        for (const auto &action : current_node.state.actions()) {
            SearchState new_state = action.execute(current_node.state);

            double heuristic_cost = compute_heuristic(new_state, *heuristic_);
            double new_cost = current_node.cost + 1;
            double total_cost = new_cost + heuristic_cost;

            const SearchState *new_state_ptr = &new_state;

            // Use 'find()' instead of 'operator[]' to avoid a type mismatch
            if (visited.find(new_state_ptr) == visited.end() || visited[new_state_ptr] > total_cost) {
                visited[new_state_ptr] = total_cost;

                auto new_actions = current_node.actions;
                new_actions.push_back(action);

                frontier.push({new_state, new_actions, total_cost});
            }
        }
    }

    return {};
}

