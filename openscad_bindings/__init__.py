from tempfile import TemporaryDirectory
import subprocess
from contextlib import contextmanager
import numpy as np
from stl.mesh import Mesh

class OpenSCADError(Exception):
    pass

def run_openscad(source_fp, output_fp):
    try:
        return subprocess.check_output([
            'openscad', '-o', 
            output_fp, source_fp]).decode()
    except subprocess.CalledProcessError as e:
        raise OpenSCADError(e.output.decode())

def scad_eval(scad_code):
    d = TemporaryDirectory()
    
    try:
        source_fp = f'{d.name}/_.scad'
        output_fp = f'{d.name}/_.stl'

        with open(source_fp, 'w') as f:
            f.write(scad_code)
        run_openscad(source_fp, output_fp)
        
        output = Mesh.from_file(output_fp)
    
    except:
        raise
        
    finally:
        d.cleanup()
        
    return output

class CodeWriter:
    def __init__(self):
        self._source = []
        self._buffer = []
        self._indent = 0

    @property
    def source(self):
        return "".join(str(x) for x in self._source)
    
    def interleave(self, inter, f, seq):
        """Call f on each item in seq, calling inter() in between."""
        seq = iter(seq)
        try:
            f(next(seq))
        except StopIteration:
            pass
        else:
            for x in seq:
                inter()
                f(x)

    def fill(self, text=""):
        """Indent a piece of text and append it, according to the current
        indentation level"""
        self.write("\n" + "    " * self._indent + text)

    def write(self, text):
        """Append a piece of text"""
        self._source.append(text)

    def buffer_writer(self, text):
        self._buffer.append(text)

    @property
    def buffer(self):
        value = "".join(self._buffer)
        self._buffer.clear()
        return value

    @contextmanager
    def block(self):
        """A context manager for preparing the source for blocks. It adds
        the character':', increases the indentation on enter and decreases
        the indentation on exit."""
        self.write("{")
        self._indent += 1
        yield
        self._indent -= 1
        self.fill('}')

    @contextmanager
    def delimit(self, start, end):
        """A context manager for preparing the source for expressions. It adds
        *start* to the buffer and enters, after exit it adds *end*."""

        self.write(start)
        yield
        self.write(end)

def format_parameter(arg):
    k, v = arg
    if v is True:
        v = 'true'
        
    if v is False:
        v = 'false'
    
    if k is not None:
        if k[0] == '_':
            k = "$" + v[1:]
    
    return f'{k}={v}' if k else v

class Model:
    def __init__(self, command_name, *args, operands=None, **kwargs):
        self.command_name = command_name
        self.args = [(None, arg) for arg in args] + list(kwargs.items())
        self.operands = operands
        
    def _write_scad_code(self, writer):
        writer.fill(self.command_name)
        
        with writer.delimit('(', ')'):
            writer.interleave(
                lambda: writer.write(', '), 
                lambda arg: writer.write(format_parameter(arg)),
                self.args)
        
        if self.operands is None:
            writer.write(';')
        else:
            with writer.block():
                for operand in self.operands:
                    operand._write_scad_code(writer)
        
    def __str__(self):
        c = CodeWriter()
        self._write_scad_code(c)
        return c.source[1:]
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        return minkowski([self, other])
    
    def union(self, other):
        return union([self, other])
    
    def intersection(self, other):
        return intersection([self, other])
    
    def difference(self, other):
        return difference([self, other])
    
    def __rmatmul__(self, other):
        return multmatrix(self, other.tolist())
    
    def render(self):
        return scad_eval(str(self))

# operators
def union(models, *args, **kwargs):
    return Model('union', *args, operands=models, **kwargs)

def intersection(models, *args, **kwargs):
    return Model('intersection', *args, operands=models, **kwargs)

def difference(models, *args, **kwargs):
    return Model('difference', *args, operands=models, **kwargs)

def hull(model, *args, **kwargs):
    return Model('hull', *args, operands=[model], **kwargs)

def minkowski(models, *args, **kwargs):
    return Model('minkowski', *args, operands=models, **kwargs)

def multmatrix(model, *args, **kwargs):
    return Model('multmatrix', *args, operands=[model], **kwargs)

def linear_extrude(model, *args, **kwargs):
    return Model('linear_extrude', *args, operands=[model], **kwargs)

def rotate_extrude(model, *args, **kwargs):
    return Model('rotate_extrude', *args, operands=[model], **kwargs)

def projection(model, *args, **kwargs):
    return Model('projection', *args, operands=[model], **kwargs)

def offset(model, *args, **kwargs):
    return Model('offset', *args, operands=[model], **kwargs)

# primitives
def cube(*args, **kwargs):
    return Model('cube', *args, **kwargs)

def sphere(*args, **kwargs):
    return Model('sphere', *args, **kwargs)

def circle(*args, **kwargs):
    return Model('circle', *args, **kwargs)

def square(*args, **kwargs):
    return Model('square', *args, **kwargs)

def polyhedron(*args, **kwargs):
    return Model('polyhedron', *args, **kwargs)

def cylinder(*args, **kwargs):
    return Model('cylinder', *args, **kwargs)

def polygon(*args, **kwargs):
    return Model('polygon', *args, **kwargs)

class Array(np.ndarray):
    def __new__(cls, data):
        return np.array(data).view(cls)
    
    def __matmul__(self, other):
        if isinstance(other, Model):
            return other.__rmatmul__(self)
        else:
            super().__matmul__(other)