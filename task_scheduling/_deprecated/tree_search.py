from collections import deque
from math import factorial


class TreeNodeBoundLo(TreeNodeBound):
    def _update_bounds(self):

        if len(self.seq_rem) == 0:
            return  # already converged

        self._bounds = [self._l_ex, self._l_ex]
        for n in self._seq_rem:  # update loss bounds
            self._bounds[0] += self._tasks[n](
                max(min(self._ch_avail), self._tasks[n].t_release)
            )

    def branch_bound(self, inplace=True, verbose=False, rng=None):

        rng = self._get_rng(rng)

        node_best = self.roll_out(inplace=False, rng=rng)  # roll-out initial solution
        stack = deque([self])  # initialize stack

        # Iterate
        while len(stack) > 0:
            node = stack.pop()  # extract node

            if len(node.seq_rem) == 0:
                if node.l_ex < node_best.l_ex:
                    node_best = node
            else:
                for node_new in node.branch(permute=True, rng=rng):
                    if node_new.l_lo < node_best.l_ex:  # new node is not dominated
                        stack.append(node_new)  # add new node to stack, LIFO

            if verbose:
                progress = 1 - sum(
                    factorial(len(node.seq_rem)) for node in stack
                ) / factorial(self.n_tasks)
                print(
                    f"Search progress: {progress:.3f}, Loss < {node_best.l_ex:.3f}",
                    end="\r",
                )
                # print(f'# Remaining Nodes = {len(stack)}, Loss <= {node_best.l_ex:.3f}', end='\r')

        if inplace:
            self.seq = node_best.seq
        else:
            return node_best
