import { useEffect, useRef } from 'react'
import { KeypointResult } from '../api/client'

// COCO skeleton connections
const SKELETON = [
  [0,1],[0,2],[1,3],[2,4],      // head
  [5,6],                         // shoulders
  [5,7],[7,9],[6,8],[8,10],      // arms
  [5,11],[6,12],[11,12],         // torso
  [11,13],[13,15],[12,14],[14,16] // legs
]

const SPINE_JOINTS = [0,5,6,11,12,13,14,15,16] // highlight spine-relevant

interface Props {
  imageUrl: string
  keypoints: KeypointResult[]
}

export default function KeypointOverlay({ imageUrl, keypoints }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!

    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.src = imageUrl
    imgRef.current = img

    img.onload = () => {
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      draw(ctx, img)
    }
  }, [imageUrl, keypoints])

  const draw = (ctx: CanvasRenderingContext2D, img: HTMLImageElement) => {
    const W = img.naturalWidth
    const H = img.naturalHeight
    ctx.clearRect(0, 0, W, H)
    ctx.drawImage(img, 0, 0)

    // Dark overlay
    ctx.fillStyle = 'rgba(10, 13, 18, 0.3)'
    ctx.fillRect(0, 0, W, H)

    const kps = keypoints.map(k => ({ x: k.x * W, y: k.y * H, name: k.name }))

    // Draw skeleton lines
    SKELETON.forEach(([a, b]) => {
      if (!kps[a] || !kps[b]) return
      const isSpine = SPINE_JOINTS.includes(a) && SPINE_JOINTS.includes(b)
      ctx.beginPath()
      ctx.moveTo(kps[a].x, kps[a].y)
      ctx.lineTo(kps[b].x, kps[b].y)
      ctx.strokeStyle = isSpine ? 'rgba(0, 212, 255, 0.85)' : 'rgba(0, 229, 160, 0.6)'
      ctx.lineWidth = isSpine ? 2.5 : 1.5
      ctx.stroke()
    })

    // Draw keypoints
    kps.forEach((kp, i) => {
      const isSpine = SPINE_JOINTS.includes(i)
      const r = isSpine ? 5 : 3.5

      // Outer glow
      ctx.beginPath()
      ctx.arc(kp.x, kp.y, r + 3, 0, Math.PI * 2)
      ctx.fillStyle = isSpine ? 'rgba(0,212,255,0.2)' : 'rgba(0,229,160,0.15)'
      ctx.fill()

      // Dot
      ctx.beginPath()
      ctx.arc(kp.x, kp.y, r, 0, Math.PI * 2)
      ctx.fillStyle = isSpine ? '#00d4ff' : '#00e5a0'
      ctx.fill()

      // Label for key joints
      if (isSpine && i !== 1 && i !== 2 && i !== 3 && i !== 4) {
        ctx.font = `bold ${Math.max(10, W * 0.022)}px JetBrains Mono`
        ctx.fillStyle = 'rgba(226,234,244,0.9)'
        ctx.fillText(kp.name, kp.x + r + 4, kp.y + 4)
      }
    })
  }

  return (
    <div className="relative w-full overflow-hidden rounded-xl bg-spine-surface border border-spine-border">
      <canvas ref={canvasRef} className="w-full h-auto" />
      {/* Legend */}
      <div className="absolute bottom-3 left-3 flex items-center gap-4 bg-spine-bg/80 backdrop-blur px-3 py-1.5 rounded-lg border border-spine-border text-xs">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-spine-accent inline-block" />
          <span className="text-spine-muted">Spine</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-spine-green inline-block" />
          <span className="text-spine-muted">Limbs</span>
        </span>
      </div>
    </div>
  )
}
