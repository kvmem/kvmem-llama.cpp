# Rebuild the native tray/EXE resources from the same vector geometry as the SVG source.
# This is a developer build tool; the installed manager has no script dependency.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$assetRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\Assets'))
New-Item -ItemType Directory -Path $assetRoot -Force | Out-Null
foreach ($design in @(
    @{Name='ninfer-active'; Background='#A4EAC3'; Foreground='#183C30'},
    @{Name='ninfer-inactive'; Background='#D0D5D3'; Foreground='#5A6460'}
)) {
    $svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect x="2" y="2" width="60" height="60" rx="16" fill="' + $design.Background + '"/><path d="M14 17H23L41 40V17H50V48H41L23 25V48H14Z" fill="' + $design.Foreground + '"/></svg>'
    [IO.File]::WriteAllText((Join-Path $assetRoot ($design.Name+'.svg')), $svg, [Text.UTF8Encoding]::new($false))
    $images = @()
    foreach ($size in @(16,20,24,32,48,64,256)) {
        $large = [Drawing.Bitmap]::new($size*4,$size*4)
        $g = [Drawing.Graphics]::FromImage($large)
        $g.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $g.ScaleTransform($size/16.0,$size/16.0)
        $round = [Drawing.Drawing2D.GraphicsPath]::new()
        $round.AddArc(2,2,32,32,180,90)
        $round.AddArc(30,2,32,32,270,90)
        $round.AddArc(30,30,32,32,0,90)
        $round.AddArc(2,30,32,32,90,90)
        $round.CloseFigure()
        $baseBrush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml($design.Background))
        $letterBrush = [Drawing.SolidBrush]::new([Drawing.ColorTranslator]::FromHtml($design.Foreground))
        $g.FillPath($baseBrush,$round)
        $points = [Drawing.PointF[]]@([Drawing.PointF]::new(14,17),[Drawing.PointF]::new(23,17),[Drawing.PointF]::new(41,40),[Drawing.PointF]::new(41,17),[Drawing.PointF]::new(50,17),[Drawing.PointF]::new(50,48),[Drawing.PointF]::new(41,48),[Drawing.PointF]::new(23,25),[Drawing.PointF]::new(23,48),[Drawing.PointF]::new(14,48))
        $g.FillPolygon($letterBrush,$points)
        $g.Dispose(); $round.Dispose(); $baseBrush.Dispose(); $letterBrush.Dispose()
        $bitmap = [Drawing.Bitmap]::new($size,$size)
        $scaled = [Drawing.Graphics]::FromImage($bitmap)
        $scaled.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $scaled.DrawImage($large,0,0,$size,$size)
        $scaled.Dispose(); $large.Dispose()
        $bytes = [IO.MemoryStream]::new()
        $bitmap.Save($bytes,[Drawing.Imaging.ImageFormat]::Png)
        if ($size -eq 64) { $bitmap.Save((Join-Path $assetRoot ($design.Name+'.png')),[Drawing.Imaging.ImageFormat]::Png) }
        $images += @{Size=$size; Bytes=$bytes.ToArray()}
        $bitmap.Dispose(); $bytes.Dispose()
    }
    $file = [IO.File]::Create((Join-Path $assetRoot ($design.Name+'.ico')))
    $writer = [IO.BinaryWriter]::new($file)
    $writer.Write([uint16]0); $writer.Write([uint16]1); $writer.Write([uint16]$images.Count)
    $offset = 6+16*$images.Count
    foreach ($entry in $images) {
        $dimension = if ($entry.Size -eq 256) {0} else {$entry.Size}
        $writer.Write([byte]$dimension); $writer.Write([byte]$dimension)
        $writer.Write([byte]0); $writer.Write([byte]0)
        $writer.Write([uint16]1); $writer.Write([uint16]32)
        $writer.Write([uint32]$entry.Bytes.Length); $writer.Write([uint32]$offset)
        $offset += $entry.Bytes.Length
    }
    foreach ($entry in $images) { $writer.Write([byte[]]$entry.Bytes) }
    $writer.Dispose(); $file.Dispose()
    Write-Output ($design.Name + ': 16 / 20 / 24 / 32 / 48 / 64 / 256 px')
}
