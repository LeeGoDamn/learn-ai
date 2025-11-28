import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

// 自定义MathJax元素标签
const customElements = [
  'mjx-container',
  'mjx-assistive-mml',
  'math',
  'maction',
  'maligngroup',
  'malignmark',
  'menclose',
  'merror',
  'mfenced',
  'mfrac',
  'mi',
  'mlongdiv',
  'mmultiscripts',
  'mn',
  'mo',
  'mover',
  'mpadded',
  'mphantom',
  'mroot',
  'mrow',
  'ms',
  'mscarries',
  'mscarry',
  'msgroup',
  'mstack',
  'msline',
  'mspace',
  'msqrt',
  'msrow',
  'mstyle',
  'msub',
  'msup',
  'msubsup',
  'mtable',
  'mtd',
  'mtext',
  'mtr',
  'munder',
  'munderover',
  'semantics',
  'annotation',
  'annotation-xml'
]

export default defineConfig({
  lang: 'zh-CN',
  title: 'AI学习之路',
  description: '面向软件工程毕业生的人工智能入门教程',
  
  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/logo.svg' }],
  ],
  
  markdown: {
    config: (md) => {
      md.use(mathjax3)
    },
    lineNumbers: true,
  },
  
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: (tag) => customElements.includes(tag)
      }
    }
  },

  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: '首页', link: '/' },
      { text: '教程', link: '/chapters/prerequisites/math' },
      { text: '资源推荐', link: '/resources' },
    ],

    sidebar: {
      '/chapters/': [
        {
          text: '第0章：前置知识',
          collapsed: false,
          items: [
            { text: '数学基础', link: '/chapters/prerequisites/math' },
            { text: 'Python科学计算', link: '/chapters/prerequisites/python-scientific' },
            { text: '数据处理基础', link: '/chapters/prerequisites/data-processing' },
          ]
        },
        {
          text: '第1章：AI基础概念',
          collapsed: false,
          items: [
            { text: '什么是人工智能', link: '/chapters/basics/what-is-ai' },
            { text: '机器学习概述', link: '/chapters/basics/ml-overview' },
            { text: '开发环境搭建', link: '/chapters/basics/environment-setup' },
          ]
        },
        {
          text: '第2章：机器学习入门',
          collapsed: false,
          items: [
            { text: '线性回归', link: '/chapters/machine-learning/linear-regression' },
          ]
        },
        {
          text: '第3章：深度学习入门',
          collapsed: false,
          items: [
            { text: '神经网络基础', link: '/chapters/deep-learning/neural-networks' },
          ]
        },
        {
          text: '第4章：实战项目',
          collapsed: false,
          items: [
            { text: '房价预测', link: '/chapters/projects/house-price' },
          ]
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/LeeGoDamn/learn-ai' }
    ],

    footer: {
      message: '面向软件工程毕业生的人工智能入门教程',
      copyright: '© 2024 AI学习之路'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: {
        dateStyle: 'short',
        timeStyle: 'medium'
      }
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})
