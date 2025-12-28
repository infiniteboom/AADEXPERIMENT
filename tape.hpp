#include "blocklist.hpp"
#include "node.hpp"
#include <cstddef>

constexpr size_t BLOCKSIZE = 16384;
constexpr size_t DATASIZE  = 65536;


class Tape
{
    BlockList<Node, BLOCKSIZE> m_nodes;
    BlockList<double, DATASIZE> m_derivatives;
    BlockList<double*, DATASIZE> m_arg_ptrs;

    char m_pad[64];

    public:
    template<size_t N>
    Node* recordNode()
    {
        Node* node = m_nodes.emplace_back(N);

        if constexpr(N)
        {
            node->m_partial_derivatives = m_derivatives.emplace_back_multi<N>();
            node->m_partial_derivatives = m_arg_ptrs.emplace_back_multi<N>();
        }

        return node;
    }

    void resetAdjoints()
    {
        for (Node& node:m_nodes)
        {
            node.m_adjoint = 0.0;
        }
    }

    void clear()
    {
        m_derivatives.clear();
        m_arg_ptrs.clear();
        m_nodes.clear();
    }

    void rewind()
    {
        m_derivatives.rewind();
        m_arg_ptrs.rewind();
        m_nodes.rewind();
    }

    void mark()
    {
        m_derivatives.setmark();
        m_arg_ptrs.setmark();
        m_nodes.setmark();
    }

    void rewindMark()
    {
        m_derivatives.rewind_to_mark();
        m_arg_ptrs.rewind_to_mark();
        m_nodes.rewind_to_mark();
    }

    using iterator = BlockList<Node, BLOCKSIZE>::iterator;

    auto begin()
    {
        return m_nodes.begin();
    }

    auto end()
    {
        return m_nodes.end();
    }

    auto markIt()
    {
        return m_nodes.mark();
    }

    auto find(Node* node)
    {
        return m_nodes.find(node);
    }
};