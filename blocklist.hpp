#include <array>
#include <cstddef>
#include <list>
#include <iterator>
#include <utility>
#include <cstring>

template <class T,size_t block_size>
class BlockList
{
    std::list<std::array<T,block_size>> m_data;
    using list_iter = decltype(m_data.begin());
    using block_iter = decltype(m_data.back().begin());

    list_iter m_current_block;
    list_iter m_last_block;
    block_iter m_next_space;
    block_iter m_last_space;

    list_iter m_marked_block;
    block_iter m_marked_space;

    private:
    void newblock()
    {
        m_data.emplace_back();

        m_current_block = m_last_block = std::prev(m_data.end());
        m_next_space    = m_current_block->begin();
        m_last_space    = m_current_block->end();
    }

    void nextblock()
    {
        if (m_current_block == m_last_block)
        {
            newblock();
        }
        else {
            ++m_current_block;
            m_next_space = m_current_block->begin();
            m_last_space = m_current_block->end();
        }
    }

    public:
    BlockList()
    {
        newblock();
    }

    void clear()
    {
        m_data.clear();
        newblock();
    }

    void rewind()
    {
        m_current_block = m_data.begin();
        m_next_space    = m_current_block->begin();
        m_last_space    = m_current_block->end();
    }

    void setmark()
    {
        m_marked_block = m_current_block;
        m_marked_space = m_next_space;
    }

    void rewind_to_mark()
    {
        m_current_block = m_marked_block;
        m_next_space    = m_marked_space;
        m_last_space    = m_current_block->end();
    }

    T* emplace_back()
    {
        if (m_next_space == m_last_space)
        {
            nextblock();
        }

        auto old_next = m_next_space;
        ++m_next_space;
        return &*old_next;
    }

    T* emplace_back_multi(const size_t n)
  {
      if (std::distance(m_next_space, m_last_space) < n)
      {
          nextblock();
      }

      auto old_next = m_next_space;
      m_next_space += n;
      return &*old_next;
  }

    template <size_t n>
    T* emplace_back_multi()
    {
        static_assert(n <= block_size, "Static check: request exceeds block size");
        return emplace_back_multi(n);
    }

    void memset(unsigned char value = 0)
    {
        for (auto& arr:m_data)
        {
            std::memset(&arr[0],value,block_size * sizeof(T));
        }
    }

    template<typename ...Args>
    T* emplace_back(Args&& ...args)
    {
        if (m_next_space == m_last_space)
        {
            nextblock();
        }

        T* emplaced = new (&*m_next_space) T(std::forward<Args>(args)...);
        ++m_next_space;
        return emplaced;
    }

    class Iterator 
    {
        list_iter  m_current_block;
        block_iter m_current_space;
        block_iter m_first_space;
        block_iter m_last_space;

        public:
        using difference_type   = std::ptrdiff_t;
        using reference         = T&;
        using pointer           = T*;
        using value_type        = T;
        using iterator_category = std::bidirectional_iterator_tag;

        Iterator() {}
        Iterator(list_iter current_block,
                block_iter current_space,
                block_iter first_space,
                block_iter last_space):
                m_current_block(current_block),
                m_current_space(current_space),
                m_first_space(first_space),
                m_last_space(last_space){}

        Iterator& operator++()
        {
            ++m_current_space;
            if (m_current_space == m_last_space)
            {
                ++m_current_block;
                m_first_space = m_current_block->begin();
                m_last_space  = m_current_block->end();
                m_current_space = m_first_space;
            }

            return *this;
        }

        Iterator& operator--()
        {
            if (m_current_space == m_first_space)
            {
                --m_current_block;
                m_first_space = m_current_block -> begin();
                m_last_space  = m_current_block -> end();
                m_current_space = m_last_space;
            }
            --m_current_space;
            return *this;
        }

        T& operator*()
        {
            return &*m_current_space;
        }

        const T& operator*() const
        {
            return *m_current_space;
        }

        T* operator->()
        {
            return &*m_current_space;
        }

        const T* operator->() const
        {
            return &*m_current_space;
        }

        bool operator==(const Iterator& rhs) const
        {
            return (m_current_block == rhs.m_current_block) &&
                    (m_current_space == rhs.m_current_space);
        }

        bool operator!=(const Iterator& rhs) const
        {
            return !(*this == rhs);
        }
    };
    using iterator = Iterator;
    iterator begin()
    {
        return iterator(m_data.begin(),m_data.begin()->begin(),
                        m_data.begin()->begin(),m_data.begin()->end());
    }

    iterator end()
    {
        auto last_block = std::prev(m_data.end());
        return iterator(m_current_block,m_next_space,m_current_block->begin(),m_current_block->end());
    }

    iterator mark()
    {
        return iterator(m_marked_block,m_marked_space,m_marked_block->begin(),m_marked_block->end());
    }

    iterator find(const T* const element)
    {
        iterator it = end();
        iterator begin = begin();

        while (it != begin)
        {
            --it;
            if (&*it == element)
            {
                return it;
            }
        }
        return end();
    }
};
