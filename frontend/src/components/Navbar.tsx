import { NavLink } from 'react-router-dom'
import { Activity, LayoutDashboard, Upload } from 'lucide-react'

export default function Navbar() {
  const linkCls = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-2 text-sm font-medium px-3 py-2 rounded-lg transition-all duration-200 ${
      isActive
        ? 'text-spine-accent bg-spine-accent/10'
        : 'text-spine-muted hover:text-spine-text hover:bg-white/5'
    }`

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 border-b border-spine-border bg-spine-bg/80 backdrop-blur-xl flex items-center px-6">
      {/* Logo */}
      <div className="flex items-center gap-2.5 mr-10">
        <div className="w-8 h-8 rounded-lg bg-spine-accent/20 border border-spine-accent/40 flex items-center justify-center glow-accent">
          <Activity size={16} className="text-spine-accent" />
        </div>
        <span className="font-display text-lg text-spine-text">SpineAI</span>
        <span className="ml-1 text-[10px] font-mono text-spine-muted border border-spine-border px-1.5 py-0.5 rounded">
          v1.0
        </span>
      </div>

      {/* Nav links */}
      <div className="flex items-center gap-1">
        <NavLink to="/" end className={linkCls}>
          <Upload size={14} />
          Analyze
        </NavLink>
        <NavLink to="/dashboard" className={linkCls}>
          <LayoutDashboard size={14} />
          Dashboard
        </NavLink>
      </div>

      {/* Right: status indicator */}
      <div className="ml-auto flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-spine-green animate-pulse" />
        <span className="text-xs text-spine-muted font-mono">Model Ready</span>
      </div>
    </nav>
  )
}
