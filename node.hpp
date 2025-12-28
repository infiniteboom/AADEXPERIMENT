#include <cstddef>
#include <vector>

struct Node 
{
    size_t  m_n;
    double  m_adjoint = 0.0;
    double* m_partial_derivatives = nullptr;   // ∂node/∂arg_i
    double** m_adjoint_targets = nullptr;      // &arg_i.adjoint()

public:
    Node(const size_t N = 0)
        : m_n(N)
    {}

    // Bind node to its local partial derivatives and adjoint target slots
    void bind(size_t n,
              double* partial_derivatives,
              double** adjoint_targets)
    {
        m_n = n;
        m_partial_derivatives = partial_derivatives;
        m_adjoint_targets = adjoint_targets;
    }

    double& adjoint()
    {
        return m_adjoint;
    }

    void propagateOne()
    {
        if (!m_n || !m_adjoint)
        {
            return;
        }

        for (size_t i = 0;i < m_n; ++i)
        {
            *(m_adjoint_targets[i]) += m_partial_derivatives[i] * m_adjoint;
        }
    }
};
