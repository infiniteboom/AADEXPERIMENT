#include <cmath>
#include <stdexcept>
#include <vector>
#include <memory>
#include <iostream>

class Node 
{
    protected:
    std::vector<std::shared_ptr<Node>> m_arguments;
    bool m_isProcessed = false;
    unsigned m_order = 0;
    double m_result = 0.0;
    double m_adjoint = 0.0;

    public:
    virtual ~Node() {};
    virtual void evaluate() = 0;
    void setOrder(unsigned order)
    {
        m_order = order;
    }

    unsigned order()
    {
        return order;
    }

    template<class T>
    void postorder(T visitFunc)
    {
        if (m_isProcessed == false)
        {
            for (auto argument:m_arguments)
            {
                argument -> postorder(visitFunc);
            }
            visitFunc(*this);
            m_isProcessed = true;
        }
    }

    double result()
    {
        return m_result;
    }

    void resetProcessed()
    {
        for (auto argument:m_arguments)
        {
            argument -> resetProcessed();
        }
        m_isProcessed = false;
    }

    double& adjoint()
    {
        return m_adjoint;
    }

    void resetAdjoints()
    {
        for (auto argument:m_arguments)
        {
            argument -> resetAdjoints();
        }
        m_adjoint = 0.0;
    }

    virtual void propagateAdjoint() = 0;
};

class PlusNode: public Node
{
    public:
    PlusNode(std::shared_ptr<Node>lhs,std::shared_ptr<Node>rhs)
    {
        m_arguments.resize(2);
        m_arguments[0] = lhs;
        m_arguments[1] = rhs;
    }

    virtual void evaluate() override
    {
        m_result = m_arguments[0] -> result() + m_arguments[1] -> result();
    }
    
    void propagteAdjoint() override
    {
        std::cout << "Propagating node " << m_order
             << " adjoint = " << m_adjoint << std::endl;

        m_arguments[0] -> adjoint() += m_adjoint;
        m_arguments[1] -> adjoint() += m_adjoint;
    }
};

class TimesNode: public Node
{
    public:
    TimesNode(std::shared_ptr<Node>lhs,std::shared_ptr<Node>rhs)
    {
        m_arguments.resize(2);
        m_arguments[0] = lhs;
        m_arguments[1] = rhs;
    };

    virtual void evaluate() override
    {
        m_result = m_arguments[0] -> result() * m_arguments[1] -> result();
    }

    void propagateAdjoint() override
    {
        std::cout << "Propagating node " << m_order
             << " adjoint = " << m_adjoint << std::endl;
        
        m_arguments[0] -> adjoint() += m_adjoint * m_arguments[1] -> result;
        m_arguments[1] -> adjoint() += m_adjoint * m_arguments[0] -> result;
    }
};


class LogNode: public Node
{
    public:
    LogNode(std::shared_ptr<Node>arg)
    {
        m_arguments.resize(1);
        m_arguments[0] = arg;
    }

    virtual void evaluate() override
    {
        m_result = std::log(m_arguments[0] -> result());
    }

    void propagateAdjoint() override
    {
        std::cout << "Propagating node " << m_order
             << " adjoint = " << m_adjoint << std::endl;
        
        m_arguments[0] -> adjoint() += m_adjoint / m_arguments[1] -> result;
    }
};

class Leaf:public Node
{
    double m_value;

    public:
    Leaf(double val):m_value((val)){}

    double value() {return m_value;}
    void setValue(double val) {m_value = val;}

    void evaluate() override
    {
        m_result = m_value;
    }

    void propagateAdjoint() override
    {
        std::cout << "Propagating node " << m_order
                    << " adjoint = " << m_adjoint << std::endl;
    }
};

class Number {
    std::shared_ptr<Node> m_node;

public:
    Number(double val) : m_node(std::make_shared<Leaf>(val)) {}

    explicit Number(std::shared_ptr<Node> node) : m_node(std::move(node)) {}

    std::shared_ptr<Node> node()
    {
        return m_node;
    }
    double value() const {
        return m_node->result(); 
    }

    void setValue(double v) {
        auto leaf = std::dynamic_pointer_cast<Leaf>(m_node);
        if (!leaf) throw std::runtime_error("不能修改公式的值，只能修改输入参数的值");
        leaf->setValue(v);
    }
    
    double evaluate()
    {
        m_node -> resetProcessed();
        m_node -> postorder([](Node& n){n.evaluate();});
        return m_node -> result();
    }

    void setOrder()
    {
        m_node -> resetProcessed();
        m_node -> postOrder([order = 0](Node& n) mutable {n.setOrder(++order);})
    }

    void propagateAdjoints()
    {
        m_node -> resetAdjoints();
        m_node -> adjoint() = 1.0;
    }
};
                      
Number operator+(Number lhs,Number rhs)
{
    return Number(std::make_shared<PlusNode>(lhs.node(),rhs.node()));
}

Number operator*(Number lhs,Number rhs)
{
    return Number(std::make_shared<TimesNode>(lhs.node(),rhs.node()));
}

Number log(Number arg)
{
    return Number(std::make_shared<LogNode>(arg.node()));
}

template<class T>
T f(T x[5])
{
    T y1 = x[2] * (5.0 * x[0] + x[1]);
    T y2 = log(y1);
    T y  = (y1 + x[3] * y2) * (y1 + y2);
    return y;
}

int main()
{
    Number x[5] = {1.0,2.0,3.0,4.0,5.0};
    Number y = f(x);
    std::cout << y.evaluate() << std::endl;
    x[0].setValue(2.5);
    y.evaluate();
    std::cout << y.value() << std::endl;
}